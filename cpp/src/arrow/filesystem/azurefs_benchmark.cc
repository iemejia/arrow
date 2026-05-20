// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

// Benchmark for Azure Blob Storage filesystem performance.
//
// This benchmark measures the impact of Azure I/O tuning changes:
//   - Block upload size (10 MiB vs 64 MiB)
//   - DownloadTo TransferOptions (SDK chunking vs single GET)
//   - I/O coalescing (hole_size_limit: 8 KB vs 12 MB)
//   - Retry policy tuning
//
// === RUNNING MODES ===
//
// 1) LOCAL (Azurite) — default, no env vars needed:
//      ./arrow-filesystem-azurefs-benchmark
//
//    Requires Azurite: npm install -g azurite
//    Starts/stops Azurite automatically.
//
// 2) REAL AZURE — set environment variables:
//      export AZURE_STORAGE_ACCOUNT="mystorageaccount"
//      export AZURE_STORAGE_KEY="base64key..."
//      ./arrow-filesystem-azurefs-benchmark
//
//    Or use a connection string:
//      export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;..."
//      ./arrow-filesystem-azurefs-benchmark
//
//    Optional: set container name (default: "arrowbench")
//      export AZURE_BENCH_CONTAINER="mycontainer"
//
//    The benchmark will create the container and test objects automatically.
//    Objects are reused across runs (only created if missing).

#include <cstdlib>
#include <future>
#include <iomanip>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"

#include "arrow/buffer.h"
#include "arrow/filesystem/azurefs.h"
#include "arrow/io/caching.h"
#include "arrow/io/interfaces.h"
#include "arrow/io/memory.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/table.h"
#include "arrow/testing/gtest_util.h"
#include "arrow/testing/process.h"
#include "arrow/testing/random.h"
#include "arrow/util/io_util.h"
#include "arrow/util/key_value_metadata.h"
#include "arrow/util/range.h"

#include "parquet/arrow/reader.h"
#include "parquet/arrow/writer.h"
#include "parquet/file_reader.h"
#include "parquet/properties.h"

namespace arrow {
namespace fs {

using ::arrow::internal::TemporaryDir;

// ---------------------------------------------------------------------------
// Helper: get optional environment variable
// ---------------------------------------------------------------------------
static std::string GetEnvOrEmpty(const char* name) {
  const char* val = std::getenv(name);
  return val ? std::string(val) : std::string();
}

static bool IsRealAzureMode() {
  return !GetEnvOrEmpty("AZURE_STORAGE_CONNECTION_STRING").empty() ||
         !GetEnvOrEmpty("AZURE_STORAGE_ACCOUNT").empty();
}

// ---------------------------------------------------------------------------
// Shared test data creation helpers
// ---------------------------------------------------------------------------

/// Create a raw blob with deterministic data
static Status MakeRawObject(AzureFileSystem* fs, const std::string& path,
                            int64_t size) {
  ARROW_ASSIGN_OR_RAISE(auto sink, fs->OpenOutputStream(path));
  const int64_t chunk_size = 1024 * 1024;
  std::string chunk(chunk_size, 'X');
  int64_t remaining = size;
  while (remaining > 0) {
    int64_t to_write = std::min(remaining, chunk_size);
    RETURN_NOT_OK(sink->Write(chunk.data(), to_write));
    remaining -= to_write;
  }
  return sink->Close();
}

/// Create a Parquet file with mixed column types
/// chunk_size controls row group size (rows per row group)
static Status MakeParquetObject(AzureFileSystem* fs, const std::string& path,
                                int num_columns, int num_rows,
                                int chunk_size = 0) {
  if (chunk_size == 0) chunk_size = num_rows;  // single row group by default
  FieldVector fields;
  fields.push_back(field("id", int64(), /*nullable=*/false));
  fields.push_back(field("timestamp", int64(), /*nullable=*/true,
                         key_value_metadata({{"min", "0"}, {"max", "10000000000"},
                                             {"null_probability", "0.01"}})));
  for (int i = 0; i < num_columns - 2; i++) {
    std::stringstream ss;
    ss << "col" << i;
    fields.push_back(
        field(ss.str(), float64(), /*nullable=*/true,
              key_value_metadata(
                  {{"min", "-1.e10"}, {"max", "1e10"}, {"null_probability", "0.05"}})));
  }
  auto batch = random::GenerateBatch(fields, num_rows, /*seed=*/42);
  ARROW_ASSIGN_OR_RAISE(auto table, Table::FromRecordBatches({batch}));

  ARROW_ASSIGN_OR_RAISE(auto sink, fs->OpenOutputStream(path));
  RETURN_NOT_OK(parquet::arrow::WriteTable(*table, default_memory_pool(), sink,
                                           /*chunk_size=*/chunk_size));
  return Status::OK();
}

/// Check if a path exists (used to skip re-creating test data on real Azure)
static bool ObjectExists(AzureFileSystem* fs, const std::string& path) {
  auto result = fs->GetFileInfo(path);
  if (!result.ok()) return false;
  return result->type() == FileType::File;
}

/// Ensure test data objects exist (creates only if missing)
static Status EnsureTestData(AzureFileSystem* fs, const std::string& container) {
  // Create container (idempotent — Azure returns success if it exists)
  RETURN_NOT_OK(fs->CreateDir(container));

  auto raw_1mib = container + "/raw_1mib";
  auto raw_10mib = container + "/raw_10mib";
  auto raw_100mib = container + "/raw_100mib";
  auto pq_c20 = container + "/pq_c20_r100k";
  auto pq_c100 = container + "/pq_c100_r50k";

  // Fabric-realistic files:
  // - 200 columns, 500K rows, 2 row groups (~250MB total, ~16MB column chunks)
  auto pq_fabric_wide = container + "/pq_fabric_c200_r500k";
  // - 200 columns, 1M rows, 4 row groups (~500MB total, ~16MB column chunks)
  auto pq_fabric_large = container + "/pq_fabric_c200_r1m";

  if (!ObjectExists(fs, raw_1mib)) {
    std::fprintf(stderr, "[bench] Creating %s ...\n", raw_1mib.c_str());
    RETURN_NOT_OK(MakeRawObject(fs, raw_1mib, 1 * 1024 * 1024));
  }
  if (!ObjectExists(fs, raw_10mib)) {
    std::fprintf(stderr, "[bench] Creating %s ...\n", raw_10mib.c_str());
    RETURN_NOT_OK(MakeRawObject(fs, raw_10mib, 10 * 1024 * 1024));
  }
  if (!ObjectExists(fs, raw_100mib)) {
    std::fprintf(stderr, "[bench] Creating %s ...\n", raw_100mib.c_str());
    RETURN_NOT_OK(MakeRawObject(fs, raw_100mib, 100 * 1024 * 1024));
  }
  if (!ObjectExists(fs, pq_c20)) {
    std::fprintf(stderr, "[bench] Creating %s ...\n", pq_c20.c_str());
    RETURN_NOT_OK(MakeParquetObject(fs, pq_c20, 20, 100000));
  }
  if (!ObjectExists(fs, pq_c100)) {
    std::fprintf(stderr, "[bench] Creating %s ...\n", pq_c100.c_str());
    RETURN_NOT_OK(MakeParquetObject(fs, pq_c100, 100, 50000));
  }
  if (!ObjectExists(fs, pq_fabric_wide)) {
    std::fprintf(stderr, "[bench] Creating %s (~250MB, 200 cols, 2 row groups)...\n",
                 pq_fabric_wide.c_str());
    // 200 cols × 500K rows × 8 bytes ≈ 800MB raw, compressed ~250MB
    // chunk_size=250000 → 2 row groups of 250K rows each
    RETURN_NOT_OK(MakeParquetObject(fs, pq_fabric_wide, 200, 500000, 250000));
  }
  if (!ObjectExists(fs, pq_fabric_large)) {
    std::fprintf(stderr, "[bench] Creating %s (~500MB, 200 cols, 4 row groups)...\n",
                 pq_fabric_large.c_str());
    // 200 cols × 1M rows × 8 bytes ≈ 1.6GB raw, compressed ~500MB
    // chunk_size=250000 → 4 row groups of 250K rows each
    RETURN_NOT_OK(MakeParquetObject(fs, pq_fabric_large, 200, 1000000, 250000));
  }

  // Multi-file dataset: 50 Parquet files × 20 cols × 50K rows (~8MB each)
  // Simulates a Delta table with 50 partition files
  auto multifile_dir = container + "/multifile_50x8mb";
  if (!ObjectExists(fs, multifile_dir + "/part_00000")) {
    std::fprintf(stderr,
                 "[bench] Creating %s (50 files × ~8MB, 20 cols × 50K rows)...\n",
                 multifile_dir.c_str());
    for (int i = 0; i < 50; i++) {
      std::stringstream ss;
      ss << multifile_dir << "/part_" << std::setfill('0') << std::setw(5) << i;
      RETURN_NOT_OK(MakeParquetObject(fs, ss.str(), 20, 50000));
    }
  }

  // Multi-file dataset: 20 Parquet files × 50 cols × 200K rows (~64MB each)
  // Simulates larger partition files from a wide table
  auto multifile_large_dir = container + "/multifile_20x64mb";
  if (!ObjectExists(fs, multifile_large_dir + "/part_00000")) {
    std::fprintf(stderr,
                 "[bench] Creating %s (20 files × ~64MB, 50 cols × 200K rows)...\n",
                 multifile_large_dir.c_str());
    for (int i = 0; i < 20; i++) {
      std::stringstream ss;
      ss << multifile_large_dir << "/part_" << std::setfill('0') << std::setw(5) << i;
      RETURN_NOT_OK(MakeParquetObject(fs, ss.str(), 50, 200000));
    }
  }

  return Status::OK();
}

// ---------------------------------------------------------------------------
// Azurite fixture: starts/stops Azurite for local benchmarks
// ---------------------------------------------------------------------------

class AzuriteFixture : public benchmark::Fixture {
 public:
  void SetUp(::benchmark::State& state) override {
    if (IsRealAzureMode()) {
      state.SkipWithMessage("Skipping Azurite tests in real Azure mode");
      return;
    }
    // Start Azurite process
    server_process_ = std::make_unique<util::Process>();
    ASSERT_OK(server_process_->SetExecutable("azurite"));
    auto temp_dir_result = TemporaryDir::Make("azurefs-bench-");
    ASSERT_OK(temp_dir_result.status());
    temp_dir_ = std::move(temp_dir_result).ValueUnsafe();
    std::vector<std::string> args = {"--silent", "--location",
                                     temp_dir_->path().ToString(),
                                     "--blobPort", "10000",
                                     "--queuePort", "10001",
                                     "--tablePort", "10002",
                                     "--skipApiVersionCheck"};
    server_process_->SetArgs(args);
    ASSERT_OK(server_process_->Execute());

    // Wait briefly for Azurite to start
    SleepFor(0.5);

    // Configure AzureOptions for Azurite
    options_.account_name = "devstoreaccount1";
    ASSERT_OK(options_.ConfigureAccountKeyCredential(
        "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/"
        "K1SZFPTOtr/KBHBeksoGMGw=="));
    // Point to local Azurite
    options_.blob_storage_authority = "127.0.0.1:10000";
    options_.dfs_storage_authority = "127.0.0.1:10000";
    options_.blob_storage_scheme = "http";
    options_.dfs_storage_scheme = "http";

    container_ = "bench";
    ASSERT_OK_AND_ASSIGN(fs_, AzureFileSystem::Make(options_));
    ASSERT_OK(EnsureTestData(fs_.get(), container_));
  }

  void TearDown(::benchmark::State& state) override {
    fs_.reset();
    if (server_process_) {
      server_process_.reset();
    }
    temp_dir_.reset();
  }

 protected:
  std::unique_ptr<util::Process> server_process_;
  std::unique_ptr<TemporaryDir> temp_dir_;
  AzureOptions options_;
  std::shared_ptr<AzureFileSystem> fs_;
  std::string container_;
};

// ---------------------------------------------------------------------------
// Real Azure fixture: connects to real Azure Blob Storage
// ---------------------------------------------------------------------------

class AzureBlobFixture : public benchmark::Fixture {
 public:
  void SetUp(::benchmark::State& state) override {
    if (!IsRealAzureMode()) {
      state.SkipWithMessage(
          "Skipping real Azure tests (set AZURE_STORAGE_ACCOUNT + AZURE_STORAGE_KEY "
          "or AZURE_STORAGE_CONNECTION_STRING)");
      return;
    }

    auto conn_string = GetEnvOrEmpty("AZURE_STORAGE_CONNECTION_STRING");
    auto account = GetEnvOrEmpty("AZURE_STORAGE_ACCOUNT");
    auto key = GetEnvOrEmpty("AZURE_STORAGE_KEY");

    if (!conn_string.empty()) {
      // Parse connection string for account name
      // Format: ...;AccountName=xxx;AccountKey=yyy;...
      auto pos = conn_string.find("AccountName=");
      if (pos != std::string::npos) {
        auto start = pos + 12;
        auto end = conn_string.find(';', start);
        options_.account_name = conn_string.substr(start, end - start);
      }
      pos = conn_string.find("AccountKey=");
      if (pos != std::string::npos) {
        auto start = pos + 11;
        auto end = conn_string.find(';', start);
        auto account_key = conn_string.substr(start, end - start);
        ASSERT_OK(options_.ConfigureAccountKeyCredential(account_key));
      }
    } else {
      options_.account_name = account;
      ASSERT_OK(options_.ConfigureAccountKeyCredential(key));
    }

    // Container name from env or default
    container_ = GetEnvOrEmpty("AZURE_BENCH_CONTAINER");
    if (container_.empty()) container_ = "arrowbench";

    ASSERT_OK_AND_ASSIGN(fs_, AzureFileSystem::Make(options_));

    // Ensure test data is present (skip re-creation if objects already exist)
    ASSERT_OK(EnsureTestData(fs_.get(), container_));
  }

  void TearDown(::benchmark::State& state) override { fs_.reset(); }

 protected:
  AzureOptions options_;
  std::shared_ptr<AzureFileSystem> fs_;
  std::string container_;
};

// ---------------------------------------------------------------------------
// Write benchmarks: measure block upload overhead
// ---------------------------------------------------------------------------

/// Write a blob
static void WriteBlob(benchmark::State& st, AzureFileSystem* fs,
                      const std::string& container, int64_t total_size,
                      int64_t write_chunk_size) {
  auto path = container + "/write_bench";
  int64_t total_bytes = 0;
  std::string chunk(write_chunk_size, 'W');
  for (auto _ : st) {
    std::shared_ptr<io::OutputStream> sink;
    ASSERT_OK_AND_ASSIGN(sink, fs->OpenOutputStream(path));
    int64_t remaining = total_size;
    while (remaining > 0) {
      int64_t to_write = std::min(remaining, write_chunk_size);
      ASSERT_OK(sink->Write(chunk.data(), to_write));
      remaining -= to_write;
    }
    ASSERT_OK(sink->Close());
    total_bytes += total_size;
  }
  st.SetBytesProcessed(total_bytes);
}

/// Read entire file in a single ReadAt call
static void ReadNaive(benchmark::State& st, AzureFileSystem* fs,
                      const std::string& path) {
  int64_t total_bytes = 0;
  for (auto _ : st) {
    ASSERT_OK_AND_ASSIGN(auto file, fs->OpenInputFile(path));
    ASSERT_OK_AND_ASSIGN(auto size, file->GetSize());
    ASSERT_OK_AND_ASSIGN(auto buf, file->ReadAt(0, size));
    total_bytes += buf->size();
  }
  st.SetBytesProcessed(total_bytes);
}

static constexpr int64_t kReadChunkSize = 5 * 1024 * 1024;  // 5 MiB

/// Read file in chunks (one HTTP GET per chunk) — worst case for cloud storage
static void ReadChunked(benchmark::State& st, AzureFileSystem* fs,
                        const std::string& path) {
  int64_t total_bytes = 0;
  for (auto _ : st) {
    ASSERT_OK_AND_ASSIGN(auto file, fs->OpenInputFile(path));
    ASSERT_OK_AND_ASSIGN(auto size, file->GetSize());
    int64_t offset = 0;
    while (offset < size) {
      int64_t to_read = std::min(kReadChunkSize, size - offset);
      ASSERT_OK_AND_ASSIGN(auto buf, file->ReadAt(offset, to_read));
      total_bytes += buf->size();
      offset += to_read;
    }
  }
  st.SetBytesProcessed(total_bytes);
}

/// Read with I/O coalescing — measures benefit of larger hole_size_limit
static void ReadCoalesced(benchmark::State& st, AzureFileSystem* fs,
                          const std::string& path, int64_t hole_size_limit,
                          int64_t range_size_limit) {
  int64_t total_bytes = 0;
  for (auto _ : st) {
    ASSERT_OK_AND_ASSIGN(auto file, fs->OpenInputFile(path));
    ASSERT_OK_AND_ASSIGN(auto size, file->GetSize());

    io::CacheOptions cache_options{hole_size_limit, range_size_limit, /*lazy=*/false,
                                   /*prefetch_limit=*/0};
    io::internal::ReadRangeCache cache(file, io::IOContext(), cache_options);

    // Simulate reading columns with gaps between them
    std::vector<io::ReadRange> ranges;
    int64_t offset = 0;
    int64_t gap = 512 * 1024;  // 512 KiB gaps between ranges (typical column gaps)
    while (offset < size) {
      int64_t to_read = std::min(kReadChunkSize, size - offset);
      ranges.push_back(io::ReadRange{offset, to_read});
      offset += to_read + gap;
    }
    ASSERT_OK(cache.Cache(ranges));

    for (const auto& range : ranges) {
      ASSERT_OK_AND_ASSIGN(auto buf, cache.Read(range));
      total_bytes += buf->size();
    }
  }
  st.SetBytesProcessed(total_bytes);
  st.counters["hole_size_kib"] = static_cast<double>(hole_size_limit) / 1024;
  st.counters["num_ranges"] =
      static_cast<double>(100 * 1024 * 1024) / (kReadChunkSize + 512 * 1024);
}

/// Parquet end-to-end read
static void ParquetRead(benchmark::State& st, AzureFileSystem* fs,
                        const std::string& path, const std::vector<int>& column_indices,
                        bool pre_buffer, const io::CacheOptions& cache_options) {
  int64_t total_bytes = 0;
  for (auto _ : st) {
    ASSERT_OK_AND_ASSIGN(auto file, fs->OpenInputFile(path));
    ASSERT_OK_AND_ASSIGN(auto size, file->GetSize());

    parquet::ArrowReaderProperties properties;
    properties.set_use_threads(true);
    properties.set_pre_buffer(pre_buffer);
    properties.set_cache_options(cache_options);

    parquet::ReaderProperties parquet_properties = parquet::default_reader_properties();
    std::unique_ptr<parquet::arrow::FileReader> reader;
    parquet::arrow::FileReaderBuilder builder;
    ASSERT_OK(builder.Open(file, parquet_properties));
    ASSERT_OK(builder.properties(properties)->Build(&reader));
    ASSERT_OK_AND_ASSIGN(auto table, reader->ReadTable(column_indices));

    total_bytes += size;
  }
  st.SetBytesProcessed(total_bytes);
}

/// Parquet write
static void ParquetWrite(benchmark::State& st, AzureFileSystem* fs,
                         const std::string& path, int num_columns, int num_rows,
                         parquet::Compression::type compression) {
  // Pre-generate the table (outside the benchmark loop)
  FieldVector fields;
  fields.push_back(field("id", int64(), /*nullable=*/false));
  for (int i = 0; i < num_columns - 1; i++) {
    std::stringstream ss;
    ss << "col" << i;
    fields.push_back(
        field(ss.str(), float64(), /*nullable=*/true,
              key_value_metadata(
                  {{"min", "-1.e10"}, {"max", "1e10"}, {"null_probability", "0.05"}})));
  }
  auto batch = random::GenerateBatch(fields, num_rows, /*seed=*/42);
  std::shared_ptr<Table> table;
  ASSERT_OK_AND_ASSIGN(table, Table::FromRecordBatches({batch}));

  int64_t total_bytes = 0;
  for (auto _ : st) {
    ASSERT_OK_AND_ASSIGN(auto sink, fs->OpenOutputStream(path));
    auto writer_props = parquet::WriterProperties::Builder()
                            .compression(compression)
                            ->build();
    ASSERT_OK(parquet::arrow::WriteTable(*table, default_memory_pool(), sink,
                                         /*chunk_size=*/num_rows, writer_props));
    total_bytes += num_rows * num_columns * 8;  // approximate
  }
  st.SetBytesProcessed(total_bytes);
}

// ---------------------------------------------------------------------------
// Macro: define + register the same benchmark for both Azurite and AzureBlob
// ---------------------------------------------------------------------------

// Azurite (local) benchmarks

BENCHMARK_DEFINE_F(AzuriteFixture, Write100MiB)(benchmark::State& st) {
  WriteBlob(st, fs_.get(), container_, 100LL * 1024 * 1024, 4 * 1024 * 1024);
}
BENCHMARK_REGISTER_F(AzuriteFixture, Write100MiB)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_DEFINE_F(AzuriteFixture, Write500MiB)(benchmark::State& st) {
  WriteBlob(st, fs_.get(), container_, 500LL * 1024 * 1024, 4 * 1024 * 1024);
}
BENCHMARK_REGISTER_F(AzuriteFixture, Write500MiB)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->Iterations(3);

BENCHMARK_DEFINE_F(AzuriteFixture, ReadNaive1MiB)(benchmark::State& st) {
  ReadNaive(st, fs_.get(), container_ + "/raw_1mib");
}
BENCHMARK_REGISTER_F(AzuriteFixture, ReadNaive1MiB)->UseRealTime();

BENCHMARK_DEFINE_F(AzuriteFixture, ReadNaive10MiB)(benchmark::State& st) {
  ReadNaive(st, fs_.get(), container_ + "/raw_10mib");
}
BENCHMARK_REGISTER_F(AzuriteFixture, ReadNaive10MiB)->UseRealTime();

BENCHMARK_DEFINE_F(AzuriteFixture, ReadNaive100MiB)(benchmark::State& st) {
  ReadNaive(st, fs_.get(), container_ + "/raw_100mib");
}
BENCHMARK_REGISTER_F(AzuriteFixture, ReadNaive100MiB)->UseRealTime();

BENCHMARK_DEFINE_F(AzuriteFixture, ReadChunked100MiB)(benchmark::State& st) {
  ReadChunked(st, fs_.get(), container_ + "/raw_100mib");
}
BENCHMARK_REGISTER_F(AzuriteFixture, ReadChunked100MiB)->UseRealTime();

BENCHMARK_DEFINE_F(AzuriteFixture, ReadCoalesced100MiB_Hole8KiB)(benchmark::State& st) {
  ReadCoalesced(st, fs_.get(), container_ + "/raw_100mib", 8192, 32 * 1024 * 1024);
}
BENCHMARK_REGISTER_F(AzuriteFixture, ReadCoalesced100MiB_Hole8KiB)->UseRealTime();

BENCHMARK_DEFINE_F(AzuriteFixture, ReadCoalesced100MiB_Hole1MiB)(benchmark::State& st) {
  ReadCoalesced(st, fs_.get(), container_ + "/raw_100mib", 1024 * 1024,
                64 * 1024 * 1024);
}
BENCHMARK_REGISTER_F(AzuriteFixture, ReadCoalesced100MiB_Hole1MiB)->UseRealTime();

BENCHMARK_DEFINE_F(AzuriteFixture, ParquetRead20Col_DefaultCache)(benchmark::State& st) {
  std::vector<int> cols(20);
  std::iota(cols.begin(), cols.end(), 0);
  ParquetRead(st, fs_.get(), container_ + "/pq_c20_r100k", cols, true,
              io::CacheOptions::LazyDefaults());
}
BENCHMARK_REGISTER_F(AzuriteFixture, ParquetRead20Col_DefaultCache)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_DEFINE_F(AzuriteFixture, ParquetRead100Col_Sparse_DefaultCache)
(benchmark::State& st) {
  std::vector<int> cols;
  for (int i = 0; i < 100; i += 10) cols.push_back(i);
  ParquetRead(st, fs_.get(), container_ + "/pq_c100_r50k", cols, true,
              io::CacheOptions::LazyDefaults());
}
BENCHMARK_REGISTER_F(AzuriteFixture, ParquetRead100Col_Sparse_DefaultCache)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_DEFINE_F(AzuriteFixture, ParquetRead100Col_Contiguous_DefaultCache)
(benchmark::State& st) {
  std::vector<int> cols(20);
  std::iota(cols.begin(), cols.end(), 0);
  ParquetRead(st, fs_.get(), container_ + "/pq_c100_r50k", cols, true,
              io::CacheOptions::LazyDefaults());
}
BENCHMARK_REGISTER_F(AzuriteFixture, ParquetRead100Col_Contiguous_DefaultCache)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_DEFINE_F(AzuriteFixture, ParquetWrite20Col_Snappy)(benchmark::State& st) {
  ParquetWrite(st, fs_.get(), container_ + "/pq_write_snappy", 20, 250000,
               parquet::Compression::SNAPPY);
}
BENCHMARK_REGISTER_F(AzuriteFixture, ParquetWrite20Col_Snappy)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_DEFINE_F(AzuriteFixture, ParquetWrite20Col_Zstd)(benchmark::State& st) {
  ParquetWrite(st, fs_.get(), container_ + "/pq_write_zstd", 20, 250000,
               parquet::Compression::ZSTD);
}
BENCHMARK_REGISTER_F(AzuriteFixture, ParquetWrite20Col_Zstd)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_DEFINE_F(AzuriteFixture, ParquetWrite100Col_Snappy)(benchmark::State& st) {
  ParquetWrite(st, fs_.get(), container_ + "/pq_write_100col_snappy", 100, 100000,
               parquet::Compression::SNAPPY);
}
BENCHMARK_REGISTER_F(AzuriteFixture, ParquetWrite100Col_Snappy)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->Iterations(3);

// ---------------------------------------------------------------------------
// Real Azure benchmarks (same tests, AzureBlobFixture)
// ---------------------------------------------------------------------------

BENCHMARK_DEFINE_F(AzureBlobFixture, Write100MiB)(benchmark::State& st) {
  WriteBlob(st, fs_.get(), container_, 100LL * 1024 * 1024, 4 * 1024 * 1024);
}
BENCHMARK_REGISTER_F(AzureBlobFixture, Write100MiB)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->Iterations(5);

BENCHMARK_DEFINE_F(AzureBlobFixture, Write500MiB)(benchmark::State& st) {
  WriteBlob(st, fs_.get(), container_, 500LL * 1024 * 1024, 4 * 1024 * 1024);
}
BENCHMARK_REGISTER_F(AzureBlobFixture, Write500MiB)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->Iterations(3);

BENCHMARK_DEFINE_F(AzureBlobFixture, ReadNaive1MiB)(benchmark::State& st) {
  ReadNaive(st, fs_.get(), container_ + "/raw_1mib");
}
BENCHMARK_REGISTER_F(AzureBlobFixture, ReadNaive1MiB)
    ->UseRealTime()
    ->Iterations(10);

BENCHMARK_DEFINE_F(AzureBlobFixture, ReadNaive10MiB)(benchmark::State& st) {
  ReadNaive(st, fs_.get(), container_ + "/raw_10mib");
}
BENCHMARK_REGISTER_F(AzureBlobFixture, ReadNaive10MiB)
    ->UseRealTime()
    ->Iterations(5);

BENCHMARK_DEFINE_F(AzureBlobFixture, ReadNaive100MiB)(benchmark::State& st) {
  ReadNaive(st, fs_.get(), container_ + "/raw_100mib");
}
BENCHMARK_REGISTER_F(AzureBlobFixture, ReadNaive100MiB)
    ->UseRealTime()
    ->Iterations(3);

BENCHMARK_DEFINE_F(AzureBlobFixture, ReadChunked100MiB)(benchmark::State& st) {
  ReadChunked(st, fs_.get(), container_ + "/raw_100mib");
}
BENCHMARK_REGISTER_F(AzureBlobFixture, ReadChunked100MiB)
    ->UseRealTime()
    ->Iterations(3);

BENCHMARK_DEFINE_F(AzureBlobFixture, ReadCoalesced100MiB_Hole8KiB)
(benchmark::State& st) {
  ReadCoalesced(st, fs_.get(), container_ + "/raw_100mib", 8192, 32 * 1024 * 1024);
}
BENCHMARK_REGISTER_F(AzureBlobFixture, ReadCoalesced100MiB_Hole8KiB)
    ->UseRealTime()
    ->Iterations(3);

BENCHMARK_DEFINE_F(AzureBlobFixture, ReadCoalesced100MiB_Hole1MiB)
(benchmark::State& st) {
  ReadCoalesced(st, fs_.get(), container_ + "/raw_100mib", 1024 * 1024,
                64 * 1024 * 1024);
}
BENCHMARK_REGISTER_F(AzureBlobFixture, ReadCoalesced100MiB_Hole1MiB)
    ->UseRealTime()
    ->Iterations(3);

BENCHMARK_DEFINE_F(AzureBlobFixture, ParquetRead20Col_DefaultCache)
(benchmark::State& st) {
  std::vector<int> cols(20);
  std::iota(cols.begin(), cols.end(), 0);
  ParquetRead(st, fs_.get(), container_ + "/pq_c20_r100k", cols, true,
              io::CacheOptions::LazyDefaults());
}
BENCHMARK_REGISTER_F(AzureBlobFixture, ParquetRead20Col_DefaultCache)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->Iterations(5);

BENCHMARK_DEFINE_F(AzureBlobFixture, ParquetRead100Col_Sparse_DefaultCache)
(benchmark::State& st) {
  std::vector<int> cols;
  for (int i = 0; i < 100; i += 10) cols.push_back(i);
  ParquetRead(st, fs_.get(), container_ + "/pq_c100_r50k", cols, true,
              io::CacheOptions::LazyDefaults());
}
BENCHMARK_REGISTER_F(AzureBlobFixture, ParquetRead100Col_Sparse_DefaultCache)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->Iterations(5);

BENCHMARK_DEFINE_F(AzureBlobFixture, ParquetRead100Col_Contiguous_DefaultCache)
(benchmark::State& st) {
  std::vector<int> cols(20);
  std::iota(cols.begin(), cols.end(), 0);
  ParquetRead(st, fs_.get(), container_ + "/pq_c100_r50k", cols, true,
              io::CacheOptions::LazyDefaults());
}
BENCHMARK_REGISTER_F(AzureBlobFixture, ParquetRead100Col_Contiguous_DefaultCache)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->Iterations(5);

BENCHMARK_DEFINE_F(AzureBlobFixture, ParquetWrite20Col_Snappy)(benchmark::State& st) {
  ParquetWrite(st, fs_.get(), container_ + "/pq_write_snappy", 20, 250000,
               parquet::Compression::SNAPPY);
}
BENCHMARK_REGISTER_F(AzureBlobFixture, ParquetWrite20Col_Snappy)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->Iterations(3);

BENCHMARK_DEFINE_F(AzureBlobFixture, ParquetWrite20Col_Zstd)(benchmark::State& st) {
  ParquetWrite(st, fs_.get(), container_ + "/pq_write_zstd", 20, 250000,
               parquet::Compression::ZSTD);
}
BENCHMARK_REGISTER_F(AzureBlobFixture, ParquetWrite20Col_Zstd)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->Iterations(3);

BENCHMARK_DEFINE_F(AzureBlobFixture, ParquetWrite100Col_Snappy)(benchmark::State& st) {
  ParquetWrite(st, fs_.get(), container_ + "/pq_write_100col_snappy", 100, 100000,
               parquet::Compression::SNAPPY);
}
BENCHMARK_REGISTER_F(AzureBlobFixture, ParquetWrite100Col_Snappy)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->Iterations(3);

// ---------------------------------------------------------------------------
// Fabric-Realistic benchmarks (Real Azure only)
// ---------------------------------------------------------------------------
// These simulate real Fabric/Spark read patterns:
//   - Wide tables (200 columns)
//   - Large files (250-500 MB) with multiple row groups
//   - Selective projection (15 cols out of 200 — typical analytics query)
//   - Sparse column selection (every 10th column — worst case for coalescing)
//   - Footer-only read (metadata discovery pattern)
// ---------------------------------------------------------------------------

// --- Fabric: Read 15 contiguous columns from 200-col, 250MB file (2 row groups) ---
// Simulates: SELECT col0..col14 FROM wide_table
// Column chunks are ~8-16 MB each; reads ~120-240 MB of data
BENCHMARK_DEFINE_F(AzureBlobFixture, Fabric_Read15of200_Contiguous)
(benchmark::State& st) {
  std::vector<int> cols(15);
  std::iota(cols.begin(), cols.end(), 0);  // first 15 columns
  ParquetRead(st, fs_.get(), container_ + "/pq_fabric_c200_r500k", cols, true,
              io::CacheOptions::LazyDefaults());
}
BENCHMARK_REGISTER_F(AzureBlobFixture, Fabric_Read15of200_Contiguous)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->Iterations(3);

// --- Fabric: Read 15 sparse columns from 200-col, 250MB file (2 row groups) ---
// Simulates: SELECT col0, col13, col26, ... FROM wide_table (BI dashboard pattern)
// Columns are spread across the file; tests coalescing vs parallel fetch tradeoff
BENCHMARK_DEFINE_F(AzureBlobFixture, Fabric_Read15of200_Sparse)
(benchmark::State& st) {
  std::vector<int> cols;
  for (int i = 0; i < 200 && static_cast<int>(cols.size()) < 15; i += 13) {
    cols.push_back(i);
  }
  ParquetRead(st, fs_.get(), container_ + "/pq_fabric_c200_r500k", cols, true,
              io::CacheOptions::LazyDefaults());
}
BENCHMARK_REGISTER_F(AzureBlobFixture, Fabric_Read15of200_Sparse)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->Iterations(3);

// --- Fabric: Read 50 columns from 200-col, 250MB file (medium-wide query) ---
// Simulates: larger analytics query pulling ~25% of columns
BENCHMARK_DEFINE_F(AzureBlobFixture, Fabric_Read50of200_Contiguous)
(benchmark::State& st) {
  std::vector<int> cols(50);
  std::iota(cols.begin(), cols.end(), 0);  // first 50 columns
  ParquetRead(st, fs_.get(), container_ + "/pq_fabric_c200_r500k", cols, true,
              io::CacheOptions::LazyDefaults());
}
BENCHMARK_REGISTER_F(AzureBlobFixture, Fabric_Read50of200_Contiguous)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->Iterations(3);

// --- Fabric: Read 15 contiguous columns from 500MB file (4 row groups) ---
// Simulates: same query on larger Delta table partition
// Tests P2 impact when column chunks are ~16MB each across 4 row groups
BENCHMARK_DEFINE_F(AzureBlobFixture, Fabric_Read15of200_Large)
(benchmark::State& st) {
  std::vector<int> cols(15);
  std::iota(cols.begin(), cols.end(), 0);
  ParquetRead(st, fs_.get(), container_ + "/pq_fabric_c200_r1m", cols, true,
              io::CacheOptions::LazyDefaults());
}
BENCHMARK_REGISTER_F(AzureBlobFixture, Fabric_Read15of200_Large)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->Iterations(3);

// --- Fabric: Read 15 sparse columns from 500MB file (4 row groups) ---
// Worst case for P2: large column chunks with gaps between them
BENCHMARK_DEFINE_F(AzureBlobFixture, Fabric_Read15of200_Sparse_Large)
(benchmark::State& st) {
  std::vector<int> cols;
  for (int i = 0; i < 200 && static_cast<int>(cols.size()) < 15; i += 13) {
    cols.push_back(i);
  }
  ParquetRead(st, fs_.get(), container_ + "/pq_fabric_c200_r1m", cols, true,
              io::CacheOptions::LazyDefaults());
}
BENCHMARK_REGISTER_F(AzureBlobFixture, Fabric_Read15of200_Sparse_Large)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->Iterations(3);

// --- Fabric: Read ALL 200 columns from 250MB file (full scan) ---
// Simulates: SELECT * for data export or full table scan
BENCHMARK_DEFINE_F(AzureBlobFixture, Fabric_ReadAll200)
(benchmark::State& st) {
  std::vector<int> cols(200);
  std::iota(cols.begin(), cols.end(), 0);
  ParquetRead(st, fs_.get(), container_ + "/pq_fabric_c200_r500k", cols, true,
              io::CacheOptions::LazyDefaults());
}
BENCHMARK_REGISTER_F(AzureBlobFixture, Fabric_ReadAll200)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->Iterations(3);

// --- Fabric: Write 200-col table with Snappy (realistic Spark output) ---
// 200 cols × 250K rows → ~250MB file with 1 row group
BENCHMARK_DEFINE_F(AzureBlobFixture, Fabric_Write200Col_Snappy)
(benchmark::State& st) {
  ParquetWrite(st, fs_.get(), container_ + "/pq_fabric_write_snappy", 200, 250000,
               parquet::Compression::SNAPPY);
}
BENCHMARK_REGISTER_F(AzureBlobFixture, Fabric_Write200Col_Snappy)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->Iterations(3);

// --- Fabric: Write 200-col table with Zstd (optimized storage) ---
BENCHMARK_DEFINE_F(AzureBlobFixture, Fabric_Write200Col_Zstd)
(benchmark::State& st) {
  ParquetWrite(st, fs_.get(), container_ + "/pq_fabric_write_zstd", 200, 250000,
               parquet::Compression::ZSTD);
}
BENCHMARK_REGISTER_F(AzureBlobFixture, Fabric_Write200Col_Zstd)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->Iterations(3);

// --- Fabric: Footer-only read (metadata discovery) ---
// Simulates: reading just the Parquet footer to discover schema/row groups
// This is the first I/O operation in any Parquet read and directly affected by
// retry policy (P1). Tests the 2 sequential GETs: 8 bytes + footer.
static void ParquetFooterOnly(benchmark::State& st, AzureFileSystem* fs,
                              const std::string& path) {
  for (auto _ : st) {
    ASSERT_OK_AND_ASSIGN(auto file, fs->OpenInputFile(path));

    parquet::ReaderProperties parquet_properties = parquet::default_reader_properties();
    std::unique_ptr<parquet::ParquetFileReader> reader =
        parquet::ParquetFileReader::Open(file, parquet_properties);
    auto metadata = reader->metadata();
    benchmark::DoNotOptimize(metadata->num_row_groups());
    benchmark::DoNotOptimize(metadata->num_columns());
    benchmark::DoNotOptimize(metadata->num_rows());
  }
  st.counters["num_row_groups"] = 0;  // filled below
  st.counters["num_columns"] = 0;
}

BENCHMARK_DEFINE_F(AzureBlobFixture, Fabric_FooterRead_250MB)
(benchmark::State& st) {
  ParquetFooterOnly(st, fs_.get(), container_ + "/pq_fabric_c200_r500k");
}
BENCHMARK_REGISTER_F(AzureBlobFixture, Fabric_FooterRead_250MB)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10);

BENCHMARK_DEFINE_F(AzureBlobFixture, Fabric_FooterRead_500MB)
(benchmark::State& st) {
  ParquetFooterOnly(st, fs_.get(), container_ + "/pq_fabric_c200_r1m");
}
BENCHMARK_REGISTER_F(AzureBlobFixture, Fabric_FooterRead_500MB)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10);

// --- Fabric: Read with pre_buffer=false (no coalescing, sequential reads) ---
// Simulates: naive reader or custom readers that issue individual column reads
BENCHMARK_DEFINE_F(AzureBlobFixture, Fabric_Read15of200_NoPrebuffer)
(benchmark::State& st) {
  std::vector<int> cols(15);
  std::iota(cols.begin(), cols.end(), 0);
  ParquetRead(st, fs_.get(), container_ + "/pq_fabric_c200_r500k", cols,
              /*pre_buffer=*/false, io::CacheOptions::LazyDefaults());
}
BENCHMARK_REGISTER_F(AzureBlobFixture, Fabric_Read15of200_NoPrebuffer)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->Iterations(3);

// ---------------------------------------------------------------------------
// Multi-file benchmarks (Real Azure only)
// ---------------------------------------------------------------------------
// Simulate Spark/Fabric patterns: reading/writing many files concurrently.
// This is where throttling (503/429) occurs and retry policy matters most.
// ---------------------------------------------------------------------------

/// Read N Parquet files in parallel using std::async (simulates Spark stage)
static void MultiFileParallelRead(benchmark::State& st, AzureFileSystem* fs,
                                  const std::string& dir, int num_files,
                                  const std::vector<int>& column_indices,
                                  bool pre_buffer) {
  // Build file paths
  std::vector<std::string> paths;
  paths.reserve(num_files);
  for (int i = 0; i < num_files; i++) {
    std::stringstream ss;
    ss << dir << "/part_" << std::setfill('0') << std::setw(5) << i;
    paths.push_back(ss.str());
  }

  int64_t total_bytes = 0;
  for (auto _ : st) {
    std::vector<std::future<int64_t>> futures;
    futures.reserve(num_files);

    for (const auto& path : paths) {
      futures.push_back(std::async(std::launch::async, [&fs, &path, &column_indices,
                                                        pre_buffer]() -> int64_t {
        auto file_result = fs->OpenInputFile(path);
        if (!file_result.ok()) return 0;
        auto file = file_result.MoveValueUnsafe();
        auto size_result = file->GetSize();
        if (!size_result.ok()) return 0;

        parquet::ArrowReaderProperties properties;
        properties.set_use_threads(true);
        properties.set_pre_buffer(pre_buffer);
        properties.set_cache_options(io::CacheOptions::LazyDefaults());

        parquet::ReaderProperties parquet_properties = parquet::default_reader_properties();
        std::unique_ptr<parquet::arrow::FileReader> reader;
        parquet::arrow::FileReaderBuilder builder;
        if (!builder.Open(file, parquet_properties).ok()) return 0;
        if (!builder.properties(properties)->Build(&reader).ok()) return 0;

        std::shared_ptr<Table> table;
        auto table_result = reader->ReadTable(column_indices);
        if (!table_result.ok()) return 0;
        return *size_result;
      }));
    }

    // Wait for all reads to complete, sum bytes
    for (auto& f : futures) {
      total_bytes += f.get();
    }
  }
  st.SetBytesProcessed(total_bytes);
  st.counters["num_files"] = static_cast<double>(num_files);
}

/// Read N Parquet files sequentially (measures per-file overhead)
static void MultiFileSequentialRead(benchmark::State& st, AzureFileSystem* fs,
                                    const std::string& dir, int num_files,
                                    const std::vector<int>& column_indices,
                                    bool pre_buffer) {
  std::vector<std::string> paths;
  paths.reserve(num_files);
  for (int i = 0; i < num_files; i++) {
    std::stringstream ss;
    ss << dir << "/part_" << std::setfill('0') << std::setw(5) << i;
    paths.push_back(ss.str());
  }

  int64_t total_bytes = 0;
  for (auto _ : st) {
    for (const auto& path : paths) {
      ASSERT_OK_AND_ASSIGN(auto file, fs->OpenInputFile(path));
      ASSERT_OK_AND_ASSIGN(auto size, file->GetSize());

      parquet::ArrowReaderProperties properties;
      properties.set_use_threads(true);
      properties.set_pre_buffer(pre_buffer);
      properties.set_cache_options(io::CacheOptions::LazyDefaults());

      parquet::ReaderProperties parquet_properties = parquet::default_reader_properties();
      std::unique_ptr<parquet::arrow::FileReader> reader;
      parquet::arrow::FileReaderBuilder builder;
      ASSERT_OK(builder.Open(file, parquet_properties));
      ASSERT_OK(builder.properties(properties)->Build(&reader));
      ASSERT_OK_AND_ASSIGN(auto table, reader->ReadTable(column_indices));
      total_bytes += size;
    }
  }
  st.SetBytesProcessed(total_bytes);
  st.counters["num_files"] = static_cast<double>(num_files);
}

/// Write N Parquet files sequentially (simulates Spark task outputs)
static void MultiFileWrite(benchmark::State& st, AzureFileSystem* fs,
                           const std::string& dir, int num_files, int num_columns,
                           int num_rows, parquet::Compression::type compression) {
  // Pre-generate table data (shared across all files, like Spark partitioned output)
  FieldVector fields;
  fields.push_back(field("id", int64(), /*nullable=*/false));
  for (int i = 0; i < num_columns - 1; i++) {
    std::stringstream ss;
    ss << "col" << i;
    fields.push_back(
        field(ss.str(), float64(), /*nullable=*/true,
              key_value_metadata(
                  {{"min", "-1.e10"}, {"max", "1e10"}, {"null_probability", "0.05"}})));
  }
  auto batch = random::GenerateBatch(fields, num_rows, /*seed=*/42);
  std::shared_ptr<Table> table;
  ASSERT_OK_AND_ASSIGN(table, Table::FromRecordBatches({batch}));

  int64_t total_bytes = 0;
  for (auto _ : st) {
    for (int i = 0; i < num_files; i++) {
      std::stringstream ss;
      ss << dir << "/write_" << std::setfill('0') << std::setw(5) << i;
      auto path = ss.str();

      ASSERT_OK_AND_ASSIGN(auto sink, fs->OpenOutputStream(path));
      auto writer_props =
          parquet::WriterProperties::Builder().compression(compression)->build();
      ASSERT_OK(parquet::arrow::WriteTable(*table, default_memory_pool(), sink,
                                           /*chunk_size=*/num_rows, writer_props));
      total_bytes += num_rows * num_columns * 8;  // approximate
    }
  }
  st.SetBytesProcessed(total_bytes);
  st.counters["num_files"] = static_cast<double>(num_files);
}

/// Open N files and read only their footers (metadata discovery pattern)
static void MultiFileFooterScan(benchmark::State& st, AzureFileSystem* fs,
                                const std::string& dir, int num_files) {
  std::vector<std::string> paths;
  paths.reserve(num_files);
  for (int i = 0; i < num_files; i++) {
    std::stringstream ss;
    ss << dir << "/part_" << std::setfill('0') << std::setw(5) << i;
    paths.push_back(ss.str());
  }

  int64_t files_scanned = 0;
  for (auto _ : st) {
    for (const auto& path : paths) {
      ASSERT_OK_AND_ASSIGN(auto file, fs->OpenInputFile(path));
      parquet::ReaderProperties parquet_properties = parquet::default_reader_properties();
      auto reader = parquet::ParquetFileReader::Open(file, parquet_properties);
      auto metadata = reader->metadata();
      benchmark::DoNotOptimize(metadata->num_row_groups());
      benchmark::DoNotOptimize(metadata->num_columns());
      benchmark::DoNotOptimize(metadata->num_rows());
      files_scanned++;
    }
  }
  st.counters["num_files"] = static_cast<double>(num_files);
  st.counters["files_per_sec"] =
      benchmark::Counter(static_cast<double>(files_scanned), benchmark::Counter::kIsRate);
}

// --- Multi-file: Parallel read 50 × 8MB files (all columns) ---
// Simulates: Spark stage reading all partitions of a Delta table concurrently
BENCHMARK_DEFINE_F(AzureBlobFixture, MultiFile_50x8MB_ParallelRead_AllCols)
(benchmark::State& st) {
  std::vector<int> cols(20);
  std::iota(cols.begin(), cols.end(), 0);
  MultiFileParallelRead(st, fs_.get(), container_ + "/multifile_50x8mb", 50, cols, true);
}
BENCHMARK_REGISTER_F(AzureBlobFixture, MultiFile_50x8MB_ParallelRead_AllCols)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->Iterations(3);

// --- Multi-file: Parallel read 50 × 8MB files (5 of 20 columns) ---
// Simulates: analytics query selecting subset of columns across all partitions
BENCHMARK_DEFINE_F(AzureBlobFixture, MultiFile_50x8MB_ParallelRead_5Cols)
(benchmark::State& st) {
  std::vector<int> cols = {0, 4, 8, 12, 16};
  MultiFileParallelRead(st, fs_.get(), container_ + "/multifile_50x8mb", 50, cols, true);
}
BENCHMARK_REGISTER_F(AzureBlobFixture, MultiFile_50x8MB_ParallelRead_5Cols)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->Iterations(3);

// --- Multi-file: Sequential read 50 × 8MB files (all columns) ---
// Simulates: single-threaded scan or serial partition processing
BENCHMARK_DEFINE_F(AzureBlobFixture, MultiFile_50x8MB_SequentialRead_AllCols)
(benchmark::State& st) {
  std::vector<int> cols(20);
  std::iota(cols.begin(), cols.end(), 0);
  MultiFileSequentialRead(st, fs_.get(), container_ + "/multifile_50x8mb", 50, cols,
                          true);
}
BENCHMARK_REGISTER_F(AzureBlobFixture, MultiFile_50x8MB_SequentialRead_AllCols)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->Iterations(3);

// --- Multi-file: Parallel read 20 × 64MB files (15 of 50 columns) ---
// Simulates: Spark stage with larger partitions, selective projection
BENCHMARK_DEFINE_F(AzureBlobFixture, MultiFile_20x64MB_ParallelRead_15Cols)
(benchmark::State& st) {
  std::vector<int> cols(15);
  std::iota(cols.begin(), cols.end(), 0);
  MultiFileParallelRead(st, fs_.get(), container_ + "/multifile_20x64mb", 20, cols, true);
}
BENCHMARK_REGISTER_F(AzureBlobFixture, MultiFile_20x64MB_ParallelRead_15Cols)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->Iterations(3);

// --- Multi-file: Parallel read 20 × 64MB files (all 50 columns) ---
// Simulates: full scan of wide-table partition files
BENCHMARK_DEFINE_F(AzureBlobFixture, MultiFile_20x64MB_ParallelRead_AllCols)
(benchmark::State& st) {
  std::vector<int> cols(50);
  std::iota(cols.begin(), cols.end(), 0);
  MultiFileParallelRead(st, fs_.get(), container_ + "/multifile_20x64mb", 20, cols, true);
}
BENCHMARK_REGISTER_F(AzureBlobFixture, MultiFile_20x64MB_ParallelRead_AllCols)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->Iterations(3);

// --- Multi-file: Footer scan 50 files (metadata discovery) ---
// Simulates: Delta table schema/stats read, or partition pruning scan
BENCHMARK_DEFINE_F(AzureBlobFixture, MultiFile_50x8MB_FooterScan)
(benchmark::State& st) {
  MultiFileFooterScan(st, fs_.get(), container_ + "/multifile_50x8mb", 50);
}
BENCHMARK_REGISTER_F(AzureBlobFixture, MultiFile_50x8MB_FooterScan)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->Iterations(3);

// --- Multi-file: Write 50 × 8MB files (Spark task output pattern) ---
// Simulates: 50 Spark tasks each writing one partition file
BENCHMARK_DEFINE_F(AzureBlobFixture, MultiFile_Write50x8MB_Snappy)
(benchmark::State& st) {
  MultiFileWrite(st, fs_.get(), container_ + "/multifile_write_50x8mb", 50, 20, 50000,
                 parquet::Compression::SNAPPY);
}
BENCHMARK_REGISTER_F(AzureBlobFixture, MultiFile_Write50x8MB_Snappy)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);

// --- Multi-file: Write 20 × 64MB files (larger partition output) ---
// Simulates: Spark job writing wide-table partitions
BENCHMARK_DEFINE_F(AzureBlobFixture, MultiFile_Write20x64MB_Snappy)
(benchmark::State& st) {
  MultiFileWrite(st, fs_.get(), container_ + "/multifile_write_20x64mb", 20, 50, 200000,
                 parquet::Compression::SNAPPY);
}
BENCHMARK_REGISTER_F(AzureBlobFixture, MultiFile_Write20x64MB_Snappy)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);

}  // namespace fs
}  // namespace arrow

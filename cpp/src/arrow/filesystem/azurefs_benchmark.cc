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
static Status MakeParquetObject(AzureFileSystem* fs, const std::string& path,
                                int num_columns, int num_rows) {
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
                                           /*chunk_size=*/num_rows));
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

/// Write a blob using the configured block_upload_size
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
  st.counters["block_size_mib"] =
      static_cast<double>(fs->options().block_upload_size) / (1024 * 1024);
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
  st.counters["block_size_mib"] =
      static_cast<double>(fs->options().block_upload_size) / (1024 * 1024);
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

}  // namespace fs
}  // namespace arrow

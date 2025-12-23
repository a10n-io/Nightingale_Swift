// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "Nightingale",
    platforms: [
        .iOS(.v17),
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "Nightingale",
            targets: ["Nightingale"]
        ),
        .executable(
            name: "GenerateAudio",
            targets: ["GenerateAudio"]
        ),
        .executable(
            name: "CrossValidate",
            targets: ["CrossValidate"]
        ),
        .executable(
            name: "GenerateTestSentences",
            targets: ["GenerateTestSentences"]
        ),
        .executable(
            name: "TTSServer",
            targets: ["TTSServer"]
        ),
        .executable(
            name: "TTSServerQuantized",
            targets: ["TTSServerQuantized"]
        ),
        .executable(
            name: "QuantizeAndSave",
            targets: ["QuantizeAndSave"]
        ),
        .executable(
            name: "StreamingProfile",
            targets: ["StreamingProfile"]
        ),
        .executable(
            name: "StreamingTest",
            targets: ["StreamingTest"]
        ),
        .executable(
            name: "StreamingBenchmark",
            targets: ["StreamingBenchmark"]
        ),
        .executable(
            name: "EarlyPlayback",
            targets: ["EarlyPlayback"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.14"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.2.0"),
    ],
    targets: [
        .target(
            name: "Nightingale",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/Nightingale"
        ),
        .executableTarget(
            name: "GenerateAudio",
            dependencies: ["Nightingale"],
            path: "test_scripts/GenerateAudio"
        ),
        .executableTarget(
            name: "CrossValidate",
            dependencies: ["Nightingale"],
            path: "test_scripts/CrossValidate"
        ),
        .executableTarget(
            name: "GenerateTestSentences",
            dependencies: ["Nightingale"],
            path: "test_scripts/GenerateTestSentences"
        ),
        .executableTarget(
            name: "TTSServer",
            dependencies: ["Nightingale"],
            path: "test_scripts/TTSServer"
        ),
        .executableTarget(
            name: "TTSServerQuantized",
            dependencies: ["Nightingale"],
            path: "test_scripts/TTSServerQuantized"
        ),
        .executableTarget(
            name: "QuantizeAndSave",
            dependencies: [
                "Nightingale",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "test_scripts/QuantizeAndSave"
        ),
        .executableTarget(
            name: "StreamingProfile",
            dependencies: [
                "Nightingale",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "test_scripts/StreamingProfile"
        ),
        .executableTarget(
            name: "StreamingTest",
            dependencies: [
                "Nightingale",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "test_scripts/StreamingTest"
        ),
        .executableTarget(
            name: "StreamingBenchmark",
            dependencies: [
                "Nightingale",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "test_scripts/StreamingBenchmark"
        ),
        .executableTarget(
            name: "EarlyPlayback",
            dependencies: [
                "Nightingale",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "test_scripts/EarlyPlayback"
        ),
    ]
)

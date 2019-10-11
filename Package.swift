// swift-tools-version:5.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "SwiftRT",
    products: [
        // Products define the executables and libraries produced by a package,
        // and make them visible to other packages.
        .library(name: "SwiftRT", targets: ["SwiftRT"]),
        .library(name: "CVulkan", targets: ["CVulkan"]),
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        // .package(url: /* package url */, from: "1.0.0"),
    ],
    targets: [
        .systemLibrary(name: "CVulkan",
                       path: "Libraries/Vulkan",
                       pkgConfig: "vulkan"),
        .target(name: "SwiftRT",
                dependencies: ["CVulkan"]),
        .testTarget(name: "SwiftRTTests",
                    dependencies: ["SwiftRT", "CVulkan"]),
    ]
)

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const m2v = b.dependency("model2vec", .{}).module("model2vec");
    const exe = b.addExecutable(.{
        .name = "m2vzig",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{.{ .name = "model2vec", .module = m2v }},
        }),
    });
    b.installArtifact(exe);
}

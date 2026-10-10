//! model2vec-zig under the shared timing protocol: one warm pass, then three runs
//! of at least two seconds, the best in texts a second. Batch 1 is
//! embedInto on one thread with an arena. The library has no batch API: any
//! larger batch runs every text over one thread per CPU, spawned once a pass
//! by this harness, each thread with its own arena, and is reported as
//! "batch all".
const std = @import("std");
const m2v = @import("model2vec");

const Ctx = struct {
    model: *const m2v.Model,
    texts: []const []const u8,
    out: []f32,
    gpa: std.mem.Allocator,
};

fn worker(c: *const Ctx, from: usize, step: usize) void {
    var arena = std.heap.ArenaAllocator.init(c.gpa);
    defer arena.deinit();
    const dim = c.model.dim;
    var i = from;
    while (i < c.texts.len) : (i += step) {
        c.model.embedInto(arena.allocator(), c.texts[i], c.out[i * dim ..][0..dim]) catch @panic("embed");
        _ = arena.reset(.retain_capacity);
    }
}

fn pass(gpa: std.mem.Allocator, model: *const m2v.Model, texts: []const []const u8, b: usize, threads: usize, out: []f32) !void {
    if (b == 1) {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const dim = model.dim;
        for (texts, 0..) |t, i| {
            try model.embedInto(arena.allocator(), t, out[i * dim ..][0..dim]);
            _ = arena.reset(.retain_capacity);
        }
        return;
    }
    // Threads spawned per chunk cost more than a chunk's work, so the
    // threads are spawned once a pass and share every text: the batch size
    // does not apply.
    const ctx = Ctx{ .model = model, .texts = texts, .out = out, .gpa = gpa };
    const n = @min(threads, texts.len);
    var hs: [512]std.Thread = undefined;
    for (0..n) |k| hs[k] = try std.Thread.spawn(.{}, worker, .{ &ctx, k, n });
    for (hs[0..n]) |h| h.join();
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, args[2], gpa, .unlimited);
    const parsed = try std.json.parseFromSlice([][]const u8, gpa, bytes, .{});
    const texts = parsed.value;

    const t0 = std.Io.Timestamp.now(io, .awake);
    var model = try m2v.Model.load(gpa, io, args[1]);
    const load_ns = std.Io.Timestamp.now(io, .awake).nanoseconds - t0.nanoseconds;
    std.debug.print("model2vec-zig: loaded in {d:.1} ms\n", .{@as(f64, @floatFromInt(load_ns)) / 1e6});
    const threads = @min(try std.Thread.getCpuCount(), 512);
    const out = try gpa.alloc(f32, texts.len * model.dim);

    // "-" times nothing: the run only writes the vectors.
    var it = std.mem.splitScalar(u8, args[3], ',');
    while (it.next()) |s| {
        if (std.mem.eql(u8, args[3], "-")) break;
        const b = try std.fmt.parseInt(usize, std.mem.trim(u8, s, " "), 10);
        try pass(gpa, &model, texts, b, threads, out);
        var best: f64 = 0;
        for (0..3) |_| {
            const start = std.Io.Timestamp.now(io, .awake);
            var done: usize = 0;
            var ns: i96 = 0;
            while (ns < 2 * std.time.ns_per_s) {
                try pass(gpa, &model, texts, b, threads, out);
                done += texts.len;
                ns = std.Io.Timestamp.now(io, .awake).nanoseconds - start.nanoseconds;
            }
            best = @max(best, @as(f64, @floatFromInt(done)) / (@as(f64, @floatFromInt(ns)) / 1e9));
        }
        if (b == 1) {
            std.debug.print("model2vec-zig batch 1: {d:.0} texts/s\n", .{best});
        } else {
            std.debug.print("model2vec-zig batch all ({d} threads): {d:.0} texts/s\n", .{ threads, best });
        }
    }
    if (args.len > 4) {
        // A text the library refuses is a row of NaN, which the score counts.
        var arena = std.heap.ArenaAllocator.init(gpa);
        const dim = model.dim;
        for (texts, 0..) |t, i| {
            model.embedInto(arena.allocator(), t, out[i * dim ..][0..dim]) catch @memset(out[i * dim ..][0..dim], std.math.nan(f32));
            _ = arena.reset(.retain_capacity);
        }
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args[4], .data = std.mem.sliceAsBytes(out) });
    }
}

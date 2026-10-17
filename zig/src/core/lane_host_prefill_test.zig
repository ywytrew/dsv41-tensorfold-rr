// Modifications Copyright 2026 ywytrew; licensed under Apache-2.0.
// Based on TensorFold and Bertholomus AI; see NOTICE and docs/dsv41-rr/CHANGES.md.
// Modified in dsv41-tensorfold-rr, 2026-10-17.
//! Scheduler contract tests: partial prompts cannot decode; other jobs can make progress.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const gpa = std.testing.allocator;

const Target = struct {
    fake: lanes.fake.Fake = .{ .gpa = gpa },
    vtable: lanes.backend.Backend.VTable = undefined,
    count: [3]usize = @splat(0),
    need: [3]usize = .{ 4, 9, 6 },
    released: [3]bool = @splat(false),
    trace: std.ArrayList(u32) = .empty,
    engine: ?api.Engine = null,
    mode: enum { normal, cancel, fail, full, cache } = .normal,
    decoded_during_fill: bool = false,
    serial: bool = false,

    fn backend(t: *Target) lanes.backend.Backend {
        const b = t.fake.backend();
        t.vtable = b.vtable.*;
        t.vtable.prefill_step = advance;
        t.vtable.prefill_decode_rounds = 8;
        t.vtable.prefill_serial = t.serial;
        t.vtable.verify = verify;
        t.vtable.queue = queue;
        t.vtable.release = release;
        return .{ .ptr = &t.fake, .vtable = &t.vtable };
    }
    fn of(ptr: *anyopaque) *Target {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        return @fieldParentPtr("fake", f);
    }
    fn index(s: *lanes.Stream) usize {
        return s.prompt()[0] - 1;
    }
    fn advance(ptr: *anyopaque, s: *lanes.Stream) anyerror!bool {
        const t = of(ptr);
        const i = index(s);
        if (t.serial) for (t.count, 0..) |count, other| {
            if (other != i and count > 0 and count < t.need[other] and !t.released[other]) return error.TwoPartialPrompts;
        };
        if (t.mode == .full and i == 1 and !t.released[0]) return error.ContextFull;
        if (t.count[i] > 0 and s.reuse.saved != null) return error.RestorePointerSurvivedYield;
        if (t.count[i] == 0) try t.fake.backend().prefill(s); // consumes restore before first yield
        t.count[i] += 1;
        try t.trace.append(gpa, @intCast(i * 100 + t.count[i]));
        if (i == 1 and t.count[i] == 2) switch (t.mode) {
            .cancel => t.engine.?.cancel(2),
            .fail => return error.InjectedPrefillFailure,
            else => {},
        };
        return t.count[i] == t.need[i];
    }
    fn verify(ptr: *anyopaque, wins: []const lanes.backend.Window, out: []lanes.backend.Verified) anyerror!void {
        const t = of(ptr);
        for (wins) |w| {
            const i = index(w.stream);
            if (t.count[i] != t.need[i]) return error.DecodedUnfinishedPrompt;
            if (i == 0 and t.count[1] > 0 and t.count[1] < t.need[1]) t.decoded_during_fill = true;
        }
        try t.fake.backend().verify(wins, out);
    }
    fn queue(ptr: *anyopaque, s: *lanes.Stream, feed: lanes.backend.Feed, position: u64) anyerror!u64 {
        const t = of(ptr);
        const i = index(s);
        if (t.count[i] != t.need[i]) return error.DecodedUnfinishedPrompt;
        if (i == 0 and t.count[1] > 0 and t.count[1] < t.need[1]) t.decoded_during_fill = true;
        return t.fake.backend().queue(s, feed, position);
    }
    fn release(ptr: *anyopaque, s: *lanes.Stream) void {
        const t = of(ptr);
        t.released[index(s)] = true;
        t.fake.backend().release(s);
    }
    fn deinit(t: *Target) void {
        t.fake.deinit();
        t.trace.deinit(gpa);
    }
};

const Box = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    prefilled: bool = false,
    early_token: bool = false,
    done: ?api.Reason = null,
    fn event(ptr: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ptr));
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        switch (e.*) {
            .prefilled => b.prefilled = true,
            .tokens => |ts| {
                if (!b.prefilled) b.early_token = true;
                b.tokens.appendSlice(gpa, ts) catch unreachable;
            },
            .finished => |f| b.done = f.reason,
        }
    }
    fn wait(b: *Box) !api.Reason {
        for (0..5000) |_| {
            b.mutex.lockUncancelable(std.testing.io);
            const done = b.done;
            b.mutex.unlock(std.testing.io);
            if (done) |d| return d;
            try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
        }
        return error.SchedulerTimeout;
    }
};

fn runCase(mode: @FieldType(Target, "mode"), serial: bool) !void {
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: Target = .{ .mode = mode, .serial = serial };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    const Snaps = struct {
        fn bytes(_: *anyopaque, at: u32) u64 {
            return 64 + 4 * @as(u64, at);
        }
        fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
            const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            return f.save(@ptrCast(@alignCast(owner.?)), at);
        }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
            return error.BackendRestores;
        }
        fn drop(ptr: *anyopaque, saved: pc.Saved) void {
            const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            f.drop(saved);
        }
    };
    // One snapshot only: another admission must evict A's restored entry while A is still filling.
    var store = pc.Store.init(gpa, .{ .ptr = &target.fake, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .min_gap = 4, .min_prompt = 0 }, 100);
    defer store.deinit();
    const prompts = [_][8]u32{ .{ 1, 9, 5, 2, 4, 8, 3, 6 }, .{ 2, 1, 7, 2, 3, 4, 5, 8 }, .{ 3, 4, 4, 1, 8, 5, 9, 7 } };
    if (mode == .cache) {
        var seed_host = api.LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
        seed_host.cache = &store;
        var seed_box: Box = .{};
        defer seed_box.tokens.deinit(gpa);
        const seed: api.Request = .{ .prompt = &prompts[0], .max_tokens = 1, .history_len = 6 };
        try seed_host.engine().submit(1, &seed, .{ .ctx = &seed_box, .event = Box.event });
        try seed_host.start();
        const reason = seed_box.wait();
        seed_host.stop();
        try std.testing.expectEqual(api.Reason.length, try reason);
        target.count = @splat(0);
        target.released = @splat(false);
        target.trace.clearRetainingCapacity();
    }
    var host = api.LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 3 });
    if (mode == .cache) host.cache = &store;
    const engine = host.engine();
    target.engine = engine;
    var boxes = [_]Box{ .{}, .{}, .{} };
    defer for (&boxes) |*b| b.tokens.deinit(gpa);
    var reqs: [3]api.Request = undefined;
    // Queue together before starting the owner thread, making the overlap deterministic.
    for (&reqs, &prompts, &boxes, 0..) |*r, *p, *b, i| {
        r.* = .{ .prompt = p, .max_tokens = 20, .history_len = 6 };
        try engine.submit(@intCast(i + 1), r, .{ .ctx = b, .event = Box.event });
    }
    try host.start();
    var stopped = false;
    defer if (!stopped) host.stop();
    for (&boxes, 0..) |*b, i| {
        const reason = try b.wait();
        const expected: api.Reason = if (i == 1 and mode == .cancel) .cancelled else if (i == 1 and mode == .fail) .failed else .length;
        try std.testing.expectEqual(expected, reason);
    }
    host.stop();
    stopped = true;
    for (&boxes, &prompts, 0..) |*b, *p, i| {
        try std.testing.expect(!b.early_token);
        if (i == 1 and (mode == .cancel or mode == .fail)) {
            try std.testing.expectEqual(@as(usize, 0), b.tokens.items.len);
            try std.testing.expectEqual(@as(usize, 2), target.count[i]);
            continue;
        }
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(gpa);
        try history.appendSlice(gpa, p);
        try std.testing.expectEqual(@as(usize, 20), b.tokens.items.len);
        for (b.tokens.items) |token| {
            try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), token);
            try history.append(gpa, token);
        }
    }
    try std.testing.expectEqual(@as(usize, 0), target.fake.lanes.count());
    try std.testing.expectEqual(@as(usize, 0), core.activeCount());
    if (mode == .cache) {
        try std.testing.expectEqual(@as(u64, 1), store.counts.hits);
        try std.testing.expect(store.counts.evicted >= 1);
    }
    if (mode == .normal and serial) {
        try std.testing.expect(target.decoded_during_fill);
        const trace = target.trace.items;
        try std.testing.expect(std.mem.indexOfScalar(u32, trace, 4).? < std.mem.indexOfScalar(u32, trace, 101).?);
        try std.testing.expect(std.mem.indexOfScalar(u32, trace, 109).? < std.mem.indexOfScalar(u32, trace, 201).?);
    }
    if (mode == .normal and !serial) {
        const trace = target.trace.items;
        try std.testing.expect(std.mem.indexOfScalar(u32, trace, 101).? < std.mem.indexOfScalar(u32, trace, 4).?);
        try std.testing.expect(std.mem.indexOfScalar(u32, trace, 201).? < std.mem.indexOfScalar(u32, trace, 4).?);
        try std.testing.expect(target.decoded_during_fill);
    }
}

test "partial prefills interleave and decode progresses before other prompts complete" {
    try runCase(.normal, false);
}
test "cancelling a partial prefill releases its slot without a first token" {
    try runCase(.cancel, false);
}
test "one failed partial prefill does not strand other admitted jobs" {
    try runCase(.fail, false);
}
test "pool-full request waits for partial owners to finish instead of failing or spinning" {
    try runCase(.full, false);
}
test "a resumed partial prompt reports its restore before another admission evicts that entry" {
    try runCase(.cache, false);
}

test "single pending prefill policy normal allows decode and eventual queue progress" {
    try runCase(.normal, true);
}

test "single pending prefill policy cancel allows decode and eventual queue progress" {
    try runCase(.cancel, true);
}

test "single pending prefill policy fail allows decode and eventual queue progress" {
    try runCase(.fail, true);
}

test "single pending prefill policy full allows decode and eventual queue progress" {
    try runCase(.full, true);
}

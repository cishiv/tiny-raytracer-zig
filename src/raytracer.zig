const std = @import("std");

const Vec3 = struct {
    x: f32,
    y: f32,
    z: f32,

    fn add(self: *Vec3, other: *Vec3) Vec3 {
        return Vec3{
            .x = self.x + other.x,
            .y = self.y + other.y,
            .z = self.z + other.z,
        };
    }

    fn sub(self: *Vec3, other: *Vec3) Vec3 {
        return Vec3{
            .x = self.x - other.x,
            .y = self.y - other.y,
            .z = self.z - other.z,
        };
    }

    fn scalarMul(self: *Vec3, scalar: f32) Vec3 {
        return Vec3{
            .x = self.x * scalar,
            .y = self.y * scalar,
            .z = self.z * scalar,
        };
    }

    fn cross(self: *Vec3, other: *Vec3) Vec3 {
        return Vec3{
            .x = self.y * other.z - self.z * other.y,
            .y = self.z * other.x - self.x * other.z,
            .z = self.x * other.y - self.y * other.x,
        };
    }

    fn dot(self: *Vec3, other: *Vec3) f32 {
        return self.x * other.x + self.y * other.y + self.z * other.z;
    }

    fn length(self: *Vec3) f32 {
        return std.math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z);
    }

    fn normalize(self: *Vec3) Vec3 {
        return self.div(self.length());
    }

    fn div(self: *Vec3, other: *Vec3) Vec3 {
        return Vec3{
            .x = self.x / other.x,
            .y = self.y / other.y,
            .z = self.z / other.z,
        };
    }
};
pub fn main() !void {
    std.debug.print("All your {s} are belong to us.\n", .{"codebase"});
}

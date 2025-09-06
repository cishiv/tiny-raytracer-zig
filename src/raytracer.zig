const std = @import("std");

// Vectors
const Vec4 = struct {
    x: f64,
    y: f64,
    z: f64,
    w: f64,
};

const Vec3 = struct {
    x: f64,
    y: f64,
    z: f64,

    fn add(self: Vec3, other: Vec3) Vec3 {
        return Vec3{
            .x = self.x + other.x,
            .y = self.y + other.y,
            .z = self.z + other.z,
        };
    }

    fn sub(self: Vec3, other: Vec3) Vec3 {
        return Vec3{
            .x = self.x - other.x,
            .y = self.y - other.y,
            .z = self.z - other.z,
        };
    }

    fn scalarMul(self: Vec3, scalar: f64) Vec3 {
        return Vec3{
            .x = self.x * scalar,
            .y = self.y * scalar,
            .z = self.z * scalar,
        };
    }

    fn cross(self: Vec3, other: Vec3) Vec3 {
        return Vec3{
            .x = self.y * other.z - self.z * other.y,
            .y = self.z * other.x - self.x * other.z,
            .z = self.x * other.y - self.y * other.x,
        };
    }

    fn dot(self: Vec3, other: Vec3) f64 {
        return self.x * other.x + self.y * other.y + self.z * other.z;
    }

    fn length(self: Vec3) f64 {
        return std.math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z);
    }

    fn normalize(self: Vec3) Vec3 {
        const len = self.magnitude();
        if (len > 0.0) {
            return self.scalarMul(1.0 / len);
        }
        return Vec3{
            .x = self.x,
            .y = self.y,
            .z = self.z,
        }; // the return type is not a ptr, so we must return a copy
    }

    fn div(self: Vec3, other: Vec3) Vec3 {
        return Vec3{
            .x = self.x / other.x,
            .y = self.y / other.y,
            .z = self.z / other.z,
        };
    }

    fn negate(self: Vec3) Vec3 {
        return Vec3{
            .x = -self.x,
            .y = -self.y,
            .z = -self.z,
        };
    }

    fn magnitude(self: Vec3) f64 {
        return std.math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z);
    }
};

// Objects
const BG_COLOR = Vec3{
    .x = 0.0,
    .y = 0.0,
    .z = 0.0,
};

const OBJECT_COLOR = Vec3{
    .x = 1.0,
    .y = 1.0,
    .z = 1.0,
};

const Material = struct {
    diffuseColor: Vec3,
    abledo: Vec4, // define
    specularExponent: f64,
    refractiveIndex: f64,
};

const IntersectResult = struct {
    intersects: bool,
    dist: f64,
};

const Sphere = struct {
    center: Vec3,
    radius: f64,
    mat: Material,

    fn ray_intersect(self: Sphere, origin: Vec3, direction: Vec3, t: *f64) IntersectResult {
        // origin of ray to sphere

        const L = self.center.sub(origin);

        // "how much" is the ray in the dir of the sphere?
        const TC = L.dot(direction);

        // check if its completely opposite
        if (TC < 0.0) {
            return IntersectResult{ .intersects = false, .dist = TC };
        }

        const d2 = L.dot(L) - (TC * TC);
        const r2 = self.radius * self.radius;

        if (d2 > r2) {
            return IntersectResult{ .intersects = false, .dist = t.* };
        }

        const mag = std.math.sqrt(r2 - d2);

        // intersection points
        t.* = TC - mag;
        const t1 = TC + mag;

        if (t.* < 0.0) {
            t.* = t1;
        }

        if (t.* < 0.0) {
            return IntersectResult{ .intersects = false, .dist = t.* };
        }

        return IntersectResult{ .intersects = true, .dist = t.* };
    }
};

const Light = struct {
    position: Vec3,
    intensity: f64,
};

// Scene
pub fn main() !void {
    const fov = std.math.pi / 3.0;

    const width = 1024;
    const height = 768;

    const ivory = Material{
        .diffuseColor = Vec3{ .x = 0.4, .y = 0.4, .z = 0.3 },
        .abledo = Vec4{ .x = 0.6, .y = 0.3, .z = 0.1, .w = 0.0 },
        .specularExponent = 50,
        .refractiveIndex = 1.0,
    };

    const redRubber = Material{
        .diffuseColor = Vec3{ .x = 0.3, .y = 0.1, .z = 0.1 },
        .abledo = Vec4{ .x = 0.9, .y = 0.1, .z = 0.1, .w = 0.0 },
        .specularExponent = 10,
        .refractiveIndex = 1.0,
    };

    const mirror = Material{
        .diffuseColor = Vec3{ .x = 1.0, .y = 1.0, .z = 1.0 },
        .abledo = Vec4{ .x = 0.0, .y = 10.0, .z = 0.8, .w = 0.0 },
        .specularExponent = 1425,
        .refractiveIndex = 1.0,
    };

    const glass = Material{
        .diffuseColor = Vec3{ .x = 0.6, .y = 0.7, .z = 0.8 },
        .abledo = Vec4{ .x = 0.0, .y = 0.5, .z = 0.1, .w = 0.8 },
        .specularExponent = 125,
        .refractiveIndex = 1.5,
    };

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Spheres
    var spheres: [4]Sphere = undefined;

    spheres[0] = Sphere{
        .center = Vec3{ .x = -3, .y = 0, .z = -16 },
        .radius = 2,
        .mat = ivory,
    };

    spheres[1] = Sphere{
        .center = Vec3{ .x = -1.0, .y = -1.5, .z = -12 },
        .radius = 2,
        .mat = glass,
    };

    spheres[2] = Sphere{
        .center = Vec3{ .x = 1.5, .y = -0.5, .z = -18 },
        .radius = 3,
        .mat = redRubber,
    };

    spheres[3] = Sphere{
        .center = Vec3{ .x = 7, .y = 5, .z = -18 },
        .radius = 4,
        .mat = mirror,
    };

    // Lights
    var lights: [3]Light = undefined;

    lights[0] = Light{
        .position = Vec3{ .x = -20, .y = 20, .z = 20 },
        .intensity = 4.5,
    };

    lights[1] = Light{
        .position = Vec3{ .x = 20, .y = -20, .z = -20 },
        .intensity = 1.8,
    };

    lights[2] = Light{
        .position = Vec3{ .x = 30, .y = 20, .z = 30 },
        .intensity = 1.7,
    };

    // framebuffer with size width * height
    var framebuffer: [width * height]Vec3 = undefined; // stack allocated
    const origin = Vec3{ .x = 0, .y = 0, .z = 0 };

    std.debug.print("Rendering scene...\n", .{});

    for (0..height) |y| {
        for (0..width) |x| {
            const xFloat: f64 = @floatFromInt(x);
            const xPos: f64 = xFloat + 0.5 - width / 2;
            const yFloat: f64 = @floatFromInt(y); // type inference can't work across all the disaprate ops in the calc
            const yPos = yFloat + 0.5 - height / 2;
            const zPos = -@as(f64, height) / (2.0 * std.math.tan((@as(f64, fov) / 2.0)));

            const rayDirection = Vec3{ .x = xPos, .y = yPos, .z = zPos };
            const rayDirectionNormalized = rayDirection.normalize();
            framebuffer[y + x * width] = castRay(origin, rayDirectionNormalized, &spheres, &lights, 0, allocator);
        }
    }

    std.debug.print("Framebuffer filled\n", .{});
}

fn castRay(origin: Vec3, direction: Vec3, spheres: []Sphere, lights: []Light, depth: u32, allocator: std.mem.Allocator) Vec3 {
    var point = Vec3{ .x = 0, .y = 0, .z = 0 };
    var N = Vec3{ .x = 0, .y = 0, .z = 0 };
    var mat = Material{ .diffuseColor = Vec3{ .x = 0, .y = 0, .z = 0 }, .abledo = Vec4{ .x = 0, .y = 0, .z = 0, .w = 0 }, .specularExponent = 0, .refractiveIndex = 0 };

    const intersects = sceneIntersect(origin, direction, &point, &N, &mat, spheres);

    if (intersects or depth > 4) {
        return BG_COLOR;
    }

    const p = 1E-3;

    const reflect_direction = reflect(direction, N);
    var reflect_origin: Vec3 = Vec3{ .x = 0, .y = 0, .z = 0 };

    if (reflect_direction.dot(N) < 0) {
        reflect_origin = point.sub(N.scalarMul(p));
    } else {
        reflect_origin = point.add(N.scalarMul(p));
    }

    const reflect_color = castRay(reflect_origin, reflect_direction, spheres, lights, depth + 1, allocator);

    const refract_direction = refract(direction, N, mat.refractiveIndex, 1.0);

    var refactor_origin: Vec3 = Vec3{ .x = 0, .y = 0, .z = 0 };

    if (refract_direction.dot(N) < 0) {
        refactor_origin = point.sub(N.scalarMul(p));
    } else {
        refactor_origin = point.add(N.scalarMul(p));
    }

    const refract_color = castRay(refactor_origin, refract_direction, spheres, lights, depth + 1, allocator);

    // shadows + diffuse

    var diffuse_light_intensity: f64 = 0.0;
    var specular_light_intensity: f64 = 0.0;

    var light_distance: f64 = 0.0;

    var shadow_origin: Vec3 = Vec3{ .x = 0, .y = 0, .z = 0 };

    for (lights) |light| {
        const light_direction = light.position.sub(point).normalize();
        light_distance = light.position.sub(point).magnitude();

        if (light_direction.dot(N) < 0) {
            shadow_origin = point.sub(N.scalarMul(p));
        } else {
            shadow_origin = point.add(N.scalarMul(p));
        }

        var shadow_point: Vec3 = Vec3{ .x = 0, .y = 0, .z = 0 };
        var shadow_normal: Vec3 = Vec3{ .x = 0, .y = 0, .z = 0 };
        var temp_mat: Material = Material{ .diffuseColor = Vec3{ .x = 0, .y = 0, .z = 0 }, .abledo = Vec4{ .x = 0, .y = 0, .z = 0, .w = 0 }, .specularExponent = 0, .refractiveIndex = 0 };

        const shadow_intersects = sceneIntersect(shadow_origin, light_direction, &shadow_point, &shadow_normal, &temp_mat, spheres);
        if (shadow_intersects and shadow_point.sub(shadow_origin).magnitude() < light_distance) {
            continue;
        }

        diffuse_light_intensity += light.intensity * @max(0.0, light_direction.dot(N));

        specular_light_intensity += std.math.pow(f64, @max(0.0, reflect(light_direction, N).dot(direction)), mat.specularExponent) * light.intensity;
    }

    // big math
    const lit = mat.diffuseColor.scalarMul(diffuse_light_intensity).scalarMul(mat.abledo.x);
    const sp = specular_light_intensity * mat.abledo.y;
    const specularVec = Vec3{
        .x = 1.0,
        .y = 1.0,
        .z = 1.0,
    };

    const specular = specularVec.scalarMul(sp);

    const re = reflect_color.scalarMul(mat.abledo.z);

    const refractVec = refract_color.scalarMul(mat.abledo.w);

    return lit.add(specular).add(re).add(refractVec);
}

// by casting a ray we can see which object is intersected with it
fn sceneIntersect(origin: Vec3, direction: Vec3, hit: *Vec3, N: *Vec3, mat: *Material, spheres: []Sphere) bool {
    var max_sphere_distance: f64 = std.math.floatMax(f64);

    for (spheres) |sphere| {
        var distI: f64 = 0; // FIXME: Does this need be passed as *distI?
        const intersect = sphere.ray_intersect(origin, direction, &distI);
        if (intersect.intersects) {
            max_sphere_distance = intersect.dist;
            const k = origin.add(direction.scalarMul(intersect.dist));
            hit.* = k;

            const n = k.sub(sphere.center).normalize();
            N.* = n;
            mat.diffuseColor = sphere.mat.diffuseColor;
            mat.abledo = sphere.mat.abledo;
            mat.specularExponent = sphere.mat.specularExponent;
        }
    }

    var checkerboard_distance: f64 = std.math.floatMax(f64);

    if (@abs(direction.y) > 1E-3) {
        const d = -(origin.y + 4) / direction.y;
        const pt = origin.add(direction.scalarMul(d));
        if (d > 0.0 and @abs(pt.x) < 10 and pt.z < -10 and pt.z > -30 and d < checkerboard_distance) {
            checkerboard_distance = d;
            hit.* = pt;
            N.* = Vec3{ .x = 0, .y = 1, .z = 0 };
            if (@mod(@as(i32, @intFromFloat(0.5 * hit.x + 1000)) + @as(i32, @intFromFloat(0.5 * hit.z)), 2) == 1) {
                mat.diffuseColor = Vec3{ .x = 0.3, .y = 0.3, .z = 0.3 };
            } else {
                mat.diffuseColor = Vec3{ .x = 0.3, .y = 0.2, .z = 0.1 };
            }
            mat.refractiveIndex = 1.0;
            mat.specularExponent = 50.0;
            mat.abledo = Vec4{ .x = 1.0, .y = 0.2, .z = 0.0, .w = 0.0 };
        }
    }

    return @min(max_sphere_distance, checkerboard_distance) < 1000.0;
}

// phong reflection
fn reflect(I: Vec3, N: Vec3) Vec3 {
    return I.sub(N.scalarMul(I.dot(N) * 2.0));
}

// snells law (refraction)
fn refract(I: Vec3, N: Vec3, etaT: f64, etaI: f64) Vec3 {
    const cos_i = @max(-1.0, @min(1.0, I.dot(N))) * -1.0;
    if (cos_i < 0.0) {
        // swap, the ray is inside the object
        return refract(I, N.negate(), etaI, etaT);
    }

    const eta = etaI / etaT;

    const k = 1.0 - eta * eta * (1 - cos_i * cos_i);

    if (k < 0.0) {
        return Vec3{ .x = 1.0, .y = 0.0, .z = 0.0 };
    }

    return I.scalarMul(eta).add(N.scalarMul((eta * cos_i - std.math.sqrt(k))));
}

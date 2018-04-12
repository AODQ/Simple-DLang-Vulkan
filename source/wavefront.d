module wavefront;
import util;
import std.stdio, std.range, std.array, std.conv, std.algorithm,
        std.string, std.math;
import std.stdio : writeln;
import std.exception : enforce;

float3 To_Float3 ( string[] args ) {
  import std.conv : to;
  return float3(args[0].to!float, args[1].to!float, args[2].to!float);
}
float2 To_Float2 ( string[] args ) {
  import std.conv : to;
  return float2(args[0].to!float, args[1].to!float);
}

enum TextureType { Diffuse };
immutable TextureType_max = TextureType.max+1;

struct BoundingBox {
  float3 bmin;
  float3 bmax;
  void Apply ( float3 pt ) {
    iota(0, 3).each!((it) {
      bmin.vector[it] = min(bmin.vector[it], pt.vector[it]);
      bmax.vector[it] = max(bmax.vector[it], pt.vector[it]);
    });
  }

  /// Returns if 'degenerate'
  void Clamp ( ref BoundingBox bbox, int params = 3) {
    iota(0, params).each!((it) {
      bmin.vector[it] = max(bmin.vector[it], bbox.bmin.vector[it]);
      bmax.vector[it] = min(bmax.vector[it], bbox.bmax.vector[it]);
    });
  }

  auto Triangle ( float3[] tris ) {
    BoundingBox rbox = BoundingBox(bmax, bmin);
    tris.each!(t => rbox.Apply(t));
    return rbox;
  }

  float2 Iterate ( float2 coord ) {
    coord.x += 1.0f;
    if ( coord.x >= bmax.x ) {
      coord.x = bmin.x;
      coord.y += 1.0f;
    }
    return coord;
  }

  BoundingBox opBinary(string op)(BoundingBox rhs) if ( op == "*" ) {
    return BoundingBox(bmin*rhs.bmin, bmax*rhs.bmax);
  }
  BoundingBox opBinary(string op)(float3 rhs) if ( op == "*" ) {
    return BoundingBox(bmin*rhs, bmax*rhs);
  }
}

class WavefrontObj {
  float3[] vertices;
  float3[] uv_coords;
  int3[] faces;
  int3[] uv_faces;

  bool has_uv = false;
  string[TextureType_max] textures;
  BoundingBox bbox;

  void Check_Valid ( ) {
    bool Is_Clamped(int3 face, size_t len) {
      return (face.x >= 0 && face.x < len &&
              face.y >= 0 && face.y < len &&
              face.z >= 0 && face.z < len );
    }
    void Clamp_Check ( ref int3[] arr, string label ) {
      import std.string : format;
      foreach ( f; arr )
        enforce(Is_Clamped(f, arr.length),
            "Face %s out of range of %s length %s"
            .format(f, label, arr.length));
    }
    Clamp_Check(faces, "geometry vertex");
    Clamp_Check(uv_faces, "UV vertex");
  }

  private void Apply_Line ( string line ) {
    auto data = line.split(" ");
    data = data.filter!(n => n != "").array;
    if ( data.length == 0 || data[0][0] == '#' ) return;
    switch ( data[0] ) {
      default: break;
      case "dtr_diffuse":
        has_uv = true;
        textures[TextureType.Diffuse] = data[1];
      break;
      case "v":
        auto vert = data[1..4].To_Float3;
        bbox.Apply(vert);
        vertices ~= vert;
      break;
      case "vt":
        auto coords = data[1..$];
        if ( coords.length == 2 )
          uv_coords ~= float3(coords.To_Float2, 0.0f);
        else {
          assert(coords.length == 3, "Incorrect vt length");
          uv_coords ~= coords.To_Float3;
        }
      break;
      case "f":
        auto reg = data[1..$].map!(n => n.Extrapolate_Region(vertices.length))
                               .array;
        if ( reg.length == 4 ) { // build two faces from a quad
          faces ~= int3(reg[0].vertex, reg[1].vertex, reg[2].vertex);
          faces ~= int3(reg[0].vertex, reg[2].vertex, reg[3].vertex);
        } else {
          faces ~= int3(reg[0].vertex, reg[1].vertex, reg[2].vertex);
        }
        uv_faces ~= int3(reg[0].uv, reg[1].uv, reg[2].uv);
      break;
    }
  }

  this ( string fname ) {
    bbox.bmin = float3( 1000.0f);
    bbox.bmax = float3(-1000.0f);
    writeln("Constructing wavefront object ", fname);
    File(fname).byLine.each!(n => Apply_Line(n.to!string));
    writeln("Checking validity of model");
    Check_Valid();
    writeln("Successfully loaded ", fname);
  }
}


private auto Extrapolate_Region ( string param, size_t vert_len ) {
  struct Region {
    int vertex;
    int uv = -1;
  }
  Region region;
  auto vars = param.split("/").filter!(n => n!="").array;
  region.vertex = vars[0].to!int.RObj_Face_Index(vert_len);
  if ( vars.length > 1 )
    region.uv = vars[1].to!int.RObj_Face_Index(vert_len);
  return region;
}

private auto RObj_Face_Index ( int t, size_t len ) {
  assert(t > 0, "Invalid face value of 0");
  return t < 0 ? cast(int)(len) + t : t-1;
}

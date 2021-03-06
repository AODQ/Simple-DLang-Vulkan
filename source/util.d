module util;
public import gl3n.math, gl3n.linalg;
alias float2 = vec2; alias int2 = vec2;
alias float3 = vec3; alias int3 = vec3;
alias float4 = vec4; alias int4 = vec4;
alias float4x4 = Matrix!(float, 4, 4);

// Just an array of unique values
struct Set(T) {
  private T[] buf;

  // O(n) complexity
  void opOpAssign(string op)(T rhs) if ( op == "~=" ) {
    foreach ( i; buf ) if ( i == rhs ) return;
    buf ~= rhs;
  }

  size_t length ( ) { return buf.length; }
  const(T*) ptr ( ) { return buf.ptr; }

  T opIndex(size_t idx) { return buf[idx]; }

  void opAssign(T[] rhs) {
    foreach ( i; rhs ) this.opOpAssign!"~="(i);
  }
  this ( T[] rhs ) {
    this = rhs;
  }

  int opApply ( int delegate(ref T) dg ) {
    foreach ( ref x; buf ) dg(x);
    return 0;
  }
}


ubyte[] Read_Spirv ( string filename ) {
  import std.file, std.stdio, std.algorithm;
  File file = File(filename);
  ubyte[] data;
  foreach ( buff; file.byChunk(4096) ) data ~= buff;
  return data;
}


T Normalize(T)(T vec) {
  return vec.normalized();
}

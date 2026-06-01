static float2 positions[3] = {
    float2(0.0, -0.5),
    float2(0.5, 0.5),
    float2(-0.5, 0.5)
};

static float3 colors[3] = {
    float3(1.0, 0.0, 0.0),
    float3(0.0, 1.0, 0.0),
    float3(0.0, 0.0, 1.0)
};

struct V2F {
    float4 pos : SV_POSITION;
    float3 color;
};

[shader("vertex")]
V2F vertMain(uint vid : SV_VertexID) {
    V2F o;
    o.pos = float4(positions[vid], 0.0, 1.0);
    o.color = colors[vid];
    return o;
}

[shader("fragment")]
float4 fragMain(V2F input) : SV_Target {
    return float4(input.color, 1.0);
}

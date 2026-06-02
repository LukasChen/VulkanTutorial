struct UniformBuffer {
    float4x4 mvp;
}

struct VSInput {
    float2 inPos;
    float3 inColor;
};

ConstantBuffer<UniformBuffer> ubo;

struct V2F {
    float4 pos : SV_POSITION;
    float3 color;
};

[shader("vertex")]
V2F vertMain(VSInput input) {
    V2F o;
    o.pos = mul(ubo.mvp, float4(input.inPos, 0.0, 1.0));
    o.color = input.inColor;
    return o;
}

[shader("fragment")]
float4 fragMain(V2F input) : SV_Target {
    return float4(input.color, 1.0);
}

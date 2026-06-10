static float3 lightDir = float3(0.0, 1.0, 1.0);

struct UniformBuffer {
    float4x4 modelMatrix;
    float4x4 viewProjectionMatrix;
    float3x3 normalMatrix;
};

struct VSInput {
    float3 inPos;
    float3 inNormal;
    float3 inColor;
};

ConstantBuffer<UniformBuffer> ubo;

struct V2F {
    float4 pos : SV_POSITION;
    float3 normal;
    float3 color;
};

[shader("vertex")]
V2F vertMain(VSInput input) {
    V2F o;

    o.pos = mul(ubo.viewProjectionMatrix, mul(ubo.modelMatrix, float4(input.inPos, 1.0)));
    o.normal = mul(ubo.modelMatrix, float4(input.inNormal, 1.0)).xyz;
    o.color = input.inColor;
    return o;
}

[shader("fragment")]
float4 fragMain(V2F input) : SV_Target {
    float ambient = 0.05f;
    float diffuse = max(dot(normalize(lightDir), normalize(input.normal)), 0.0);
    float lit = ambient + (1.0f - ambient) * diffuse;
    float3 color = lit * input.color;
    return float4(color, 1.0);
}

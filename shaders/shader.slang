static float3 lightDir = float3(0.0, 1.0, -1.0);

struct UniformBuffer {
    float4x4 viewProjectionMatrix;
};

struct VSInput {
    float3 inPos;
    float3 inNormal;
    float3 inColor;
    float4 instanceModel0;
    float4 instanceModel1;
    float4 instanceModel2;
    float4 instanceModel3;
    float3 instanceNormal0;
    float3 instanceNormal1;
    float3 instanceNormal2;
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

    float4x4 modelMatrix = transpose(float4x4(
        input.instanceModel0,
        input.instanceModel1,
        input.instanceModel2,
        input.instanceModel3
    ));
    float3x3 normalMatrix = transpose(float3x3(
        input.instanceNormal0,
        input.instanceNormal1,
        input.instanceNormal2
    ));

    o.pos = mul(ubo.viewProjectionMatrix, mul(modelMatrix, float4(input.inPos, 1.0)));
    o.normal = mul(normalMatrix, input.inNormal);
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

struct UniformBuffer {
    float4x4 projectionMatrix;
    float4x4 viewMatrix;
    float4x4 lightViewProjection;
    float4 lightDirection;
};

[[vk::binding(0, 0)]]
ConstantBuffer<UniformBuffer> ubo;

struct VSInput {
    float3 inPos;
    float3 inNormal;
    float2 inTexCoord;

    float4 instanceModel0;
    float4 instanceModel1;
    float4 instanceModel2;
    float4 instanceModel3;

    float3 instanceNormal0;
    float3 instanceNormal1;
    float3 instanceNormal2;
};

[shader("vertex")]
float4 vertMain(VSInput input) : SV_Position {
    float4x4 modelMatrix = transpose(float4x4(
        input.instanceModel0,
        input.instanceModel1,
        input.instanceModel2,
        input.instanceModel3
    ));

    float4 worldPosition =
        mul(modelMatrix, float4(input.inPos, 1.0));

    return mul(ubo.lightViewProjection, worldPosition);
}

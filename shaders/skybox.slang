struct UniformBuffer {
    float4x4 proj;
    float4x4 view;
};

struct VSInput {
    float3 inPos;
    float3 inNormal;
    float2 inTexCoord;
};

struct V2F {
    float4 pos : SV_POSITION;
    float2 texCoord : TEXCOORD0;
};

[[vk::binding(0, 0)]]
ConstantBuffer<UniformBuffer> ubo;
[[vk::binding(0, 1)]]
Sampler2D texture;

static const float PI = 3.14159265358979323846;

[shader("vertex")]
V2F vertMain(VSInput input) {
    V2F output;

    float4x4 viewNoTranslation = ubo.view;
    viewNoTranslation[0][3] = 0.0f;
    viewNoTranslation[1][3] = 0.0f;
    viewNoTranslation[2][3] = 0.0f;
    output.pos = mul(ubo.proj, mul(viewNoTranslation, float4(input.inPos, 1.0f)));
    output.pos = output.pos.xyww;
    output.texCoord = input.inTexCoord;
    return output;
}

[shader("fragment")]
float4 fragMain(V2F input) : SV_Target {
    return texture.Sample(input.texCoord);
}

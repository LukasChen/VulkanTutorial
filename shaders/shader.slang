// static float3 lightDir = float3(0.0, 1.0, -1.0);

struct UniformBuffer {
    float4x4 projectionMatrix;
    float4x4 viewMatrix;
    float4x4 lightViewProjection;
    float4 lightDirection;
};

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

[[vk::binding(0, 0)]]
ConstantBuffer<UniformBuffer> ubo;

[[vk::binding(1, 0)]]
Sampler2D shadowMap;

struct V2F {
    float4 pos : SV_POSITION;
    float3 normal;
    float2 texCoord;
    float4 lightClipPosition;
};

float calculateShadow(float4 lightClipPosition, float3 worldNormal) {
    if (lightClipPosition.w <= 0.0) {
        return 0.0;
    }

    float3 projected = lightClipPosition.xyz / lightClipPosition.w;
    float2 uv = projected.xy * 0.5 + 0.5;

    float currentDepth = projected.z;

    if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) {
        return 0.0;
    }

    float3 lightDirection = normalize(ubo.lightDirection.xyz);

    float normalDotLight = max(dot(worldNormal, lightDirection), 0.0);

    static const float2 texelSize = 1.0 / float2(2048.0, 2048.0);

    float shadow = 0.0;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            float2 sampledUv = uv + float2(x, y) * texelSize;
            float storedDepth = shadowMap.Sample(sampledUv).r;
            shadow += currentDepth > storedDepth ? 1.0 : 0.0;
        }
    }

    shadow /= 9.0;
    return shadow;
} 

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

    float4 worldPosition = mul(modelMatrix, float4(input.inPos, 1.0));

    o.pos = mul(ubo.projectionMatrix, mul(ubo.viewMatrix, worldPosition));
    o.lightClipPosition = mul(ubo.lightViewProjection, worldPosition);
    o.normal = mul(normalMatrix, input.inNormal);
    o.texCoord = input.inTexCoord;
    return o;
}

[[vk::binding(0, 1)]]
Sampler2D texture;

[shader("fragment")]
float4 fragMain(V2F input) : SV_Target {
    float ambient = 0.05f;
    float3 lightDir = normalize(ubo.lightDirection.xyz);

    float diffuse = max(dot(normalize(lightDir), normalize(input.normal)), 0.0);

    float shadow  = calculateShadow(input.lightClipPosition, input.normal);

    float directLighting = diffuse * (1.0 - shadow);

    float lit = ambient + (1.0f - ambient) * directLighting;

    float4 color = lit * texture.Sample(input.texCoord);
    return color;
}

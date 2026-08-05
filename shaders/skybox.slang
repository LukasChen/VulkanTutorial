struct UniformBuffer {
    float4x4 proj;
    float4x4 view;
    float4x4 lightViewProjection;
    float4 lightDirection;
};

struct VSInput {
    float3 inPos;
    float3 inNormal;
    float2 inTexCoord;
};

struct V2F {
    float4 pos : SV_POSITION;
    float2 texCoord : TEXCOORD0;
    float3 direction : TEXCOORD1;
};

[[vk::binding(0, 0)]]
ConstantBuffer<UniformBuffer> ubo;
[[vk::binding(0, 1)]]
Sampler2D texture;

static const float PI = 3.14159265358979323846;
static const float3 LightDir = normalize(float3(-3.0, 3.0, -3.0));
static const float CameraHeight = 1.000001;
static const float3 InvWaveLength = float3(5.60204474633241, 9.4732844379203038, 19.643802610477206);
static const float InnerRadius = 1.0;
static const float OuterRadius = 1.025;
static const float ESun = 10.0;
static const float Kr = 0.0025;
static const float Km = 0.0015;
static const float ScaleDepth = 0.25;
static const int Samples = 2;
static const float G = -0.99;
static const float3 GroundColor = float3(0.37, 0.35, 0.34);
static const float Gamma = 1.0 / 2.2;
static const float3 SunColor = float3(1.0, 0.93, 0.78);
static const float SunDiskCos = 0.99985;
static const float SunGlowCos = 0.992;

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
    output.direction = input.inPos;
    return output;
}

float scale(float fCos) {
    float x = 1.0 - fCos;
    return ScaleDepth * exp(-0.00287 + x * (0.459 + x * (3.83 + x * (-6.80 + x * 5.25))));
}

float2 getIntersections(float3 pos, float3 dir, float dist2, float rad2) {
    float b = 2.0 * dot(pos, dir);
    float c = dist2 - rad2;
    float det = max(0.0, b * b - 4.0 * c);
    float root = sqrt(det);
    return 0.5 * float2(-b - root, -b + root);
}

float getRayleighPhase(float fCos2) {
    return 0.75 * (2.0 + 0.5 * fCos2);
}

float getMiePhase(float fCos, float fCos2, float g, float g2) {
    return 1.5 * ((1.0 - g2) / (2.0 + g2)) * (1.0 + fCos2) /
        pow(1.0 + g2 - 2.0 * g * fCos, 1.5);
}

float3 computeAtmosphere(float3 rayDir) {
    float innerRadius2 = InnerRadius * InnerRadius;
    float outerRadius2 = OuterRadius * OuterRadius;
    float krESun = Kr * ESun;
    float kmESun = Km * ESun;
    float kr4PI = Kr * 4.0 * PI;
    float km4PI = Km * 4.0 * PI;
    float atmosphereScale = 1.0 / (OuterRadius - InnerRadius);
    float scaleOverScaleDepth = atmosphereScale / ScaleDepth;
    float g2 = G * G;

    float3 lightDir = LightDir;
    float3 rayOri = float3(0.0, CameraHeight, 0.0);
    float cameraHeight = length(rayOri);
    float cameraHeight2 = cameraHeight * cameraHeight;

    float2 innerIntersections = getIntersections(rayOri, rayDir, cameraHeight2, innerRadius2);
    float2 outerIntersections = getIntersections(rayOri, rayDir, cameraHeight2, outerRadius2);
    bool isGround = 0.0 < innerIntersections.x;

    if (outerIntersections.x == outerIntersections.y) {
        return float3(0.0);
    }

    float rayNear = max(0.0, outerIntersections.x);
    float rayFar = isGround ? innerIntersections.x : outerIntersections.y;
    float3 farPos = rayOri + rayDir * rayFar;
    float3 farPosNorm = normalize(farPos);

    float3 startPos = rayOri + rayDir * rayNear;
    float startPosHeight = length(startPos);
    float3 startPosNorm = startPos / startPosHeight;
    float startAngle = dot(rayDir, startPosNorm);
    float startDepth = exp(scaleOverScaleDepth * (InnerRadius - startPosHeight));
    float startOffset = startDepth * scale(startAngle);

    float cameraAngle = dot(-rayDir, farPosNorm);
    float cameraScale = scale(cameraAngle);
    float cameraOffset = exp((InnerRadius - cameraHeight) / ScaleDepth) * cameraScale;

    float temp = scale(dot(farPosNorm, lightDir)) + scale(dot(farPosNorm, -rayDir));

    float sampleLength = (rayFar - rayNear) / float(Samples);
    float scaledLength = sampleLength * atmosphereScale;
    float3 sampleDir = rayDir * sampleLength;
    float3 samplePoint = startPos + sampleDir * 0.5;

    float3 frontColor = float3(0.0);
    float3 attenuate = float3(0.0);
    for (int i = 0; i < Samples; ++i) {
        float height = length(samplePoint);
        float depth = exp(scaleOverScaleDepth * (InnerRadius - height));
        float lightAngle = dot(lightDir, samplePoint) / height;
        float sampleCameraAngle = dot(rayDir, samplePoint) / height;
        float scatter = isGround
            ? depth * temp - cameraOffset
            : startOffset + depth * (scale(lightAngle) - scale(sampleCameraAngle));
        attenuate = exp(-scatter * (InvWaveLength * kr4PI + km4PI));
        frontColor += attenuate * (depth * scaledLength);
        samplePoint += sampleDir;
    }

    frontColor = clamp(frontColor, 0.0, 3.0);
    float3 rayleighColor = frontColor * (InvWaveLength * krESun);
    float3 mieColor = frontColor * kmESun;

    if (isGround) {
        float3 groundMieColor = clamp(attenuate, 0.0, 3.0);
        float3 groundColor = 1.0 - exp(-(rayleighColor + GroundColor * groundMieColor));
        return pow(groundColor, float3(Gamma));
    }

    float fCos = dot(-lightDir, rayDir);
    float fCos2 = fCos * fCos;
    float3 skyColor =
        getRayleighPhase(fCos2) * rayleighColor +
        getMiePhase(fCos, fCos2, G, g2) * mieColor;
    return pow(skyColor, float3(Gamma));
}

float3 computeSun(float3 rayDir) {
    float sunCos = dot(rayDir, LightDir);
    float disk = smoothstep(SunDiskCos, 1.0, sunCos);
    float glow = smoothstep(SunGlowCos, 1.0, sunCos);
    return SunColor * (disk * 6.0 + glow * glow * 0.8);
}

[shader("fragment")]
float4 fragMain(V2F input) : SV_Target {
    float3 direction = normalize(input.direction);
    float3 color = computeAtmosphere(direction) + computeSun(direction);
    return float4(color, 1.0);
}

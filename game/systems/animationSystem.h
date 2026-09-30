#pragma once

#include <algorithm>
#include <cmath>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include "../../ecs/ecs.hpp"
#include "../../components/components_common.h"

#include "../../input.h"
#include "../../engine.h"

class AnimationSystem : public System<Animator, SkinMeshRenderer> {
public:
    using System::System;

protected:
	void update(View<Animator, SkinMeshRenderer>& view, float deltaTime) override {
        for (auto [animator, skinRenderer] : view) {
            const AnimationAsset& asset = m_engine.getResource()->getAnimationAsset(animator.assetHandle);

            const AnimationClip& clip = asset.clips[animator.clipIndex];

            if (animator.playing) {
                animator.time += deltaTime * animator.speed;
                if (animator.looping) {
                    animator.time = std::fmod(animator.time, clip.duration);
                } else {
                    animator.time = std::min(animator.time, clip.duration);
                }
            }

            std::vector<JointPose> poses;
            poses.reserve(asset.skeleton.joints.size());

            for (const SkeletonJoint& joint : asset.skeleton.joints) {
                poses.push_back({
                    joint.bindTranslation,
                    joint.bindRotation,
                    joint.bindScale
                });
            }

            for (const AnimationChannel& channel : clip.channels) {
                const glm::vec4 value = sampleChannel(channel, animator.time);

                JointPose& pose = poses[channel.joint];

                switch (channel.path) {
                    case AnimationPath::Translation:
                        pose.translate = glm::vec3(value);
                        break;
                    case AnimationPath::Rotation:
                        pose.rotation = glm::normalize(glm::quat(value.w, value.x, value.y, value.z));
                        break;
                    case AnimationPath::Scale:
                        pose.scale = glm::vec3(value.x, value.y, value.z);
                        break;
                }
            }

            std::vector<glm::mat4> globalTransforms(asset.skeleton.joints.size());

            animator.jointPalette.resize(asset.skeleton.joints.size());

            for (uint32_t jointSlot : asset.skeleton.evaluationOrder) {
                const SkeletonJoint& joint = asset.skeleton.joints[jointSlot];

                const glm::mat4 local = composeLocal(poses[jointSlot]);

                globalTransforms[jointSlot] = joint.parentJoint < 0
                    ? asset.skeleton.rootTransform * local
                    : globalTransforms[joint.parentJoint] * local;

                animator.jointPalette[jointSlot] = globalTransforms[jointSlot] * joint.inverseBind;
            }
        }
    }
private:
    static glm::vec4 sampleChannel(const AnimationChannel& channel, float time) {
        const auto& times = channel.times;
        const auto& values = channel.values;

        if (time <= times.front()) {
            return values.front();
        } 

        if (time >= times.back()) {
            return values.back();
        } 

        const auto upper = std::upper_bound(times.begin(), times.end(), time);
        const size_t next = static_cast<size_t>(upper - times.begin());

        const size_t previous = next - 1;

        if (channel.interpolation == Interpolation::Step) {
            return values[previous];
        }

        const float t = (time - times[previous]) / (times[next] - times[previous]);

        if (channel.path == AnimationPath::Rotation) {
            glm::quat from(values[previous].w, values[previous].x, values[previous].y, values[previous].z);
            glm::quat to(values[next].w, values[next].x, values[next].y, values[next].z);

            // q and -q represent the same orientation; keep interpolation on the short arc.
            if (glm::dot(from, to) < 0.0f) {
                to = -to;
            }

            const glm::quat result = glm::normalize(glm::mix(from, to, t));
            return glm::vec4(result.x, result.y, result.z, result.w);
        }

        return glm::mix(values[previous], values[next], t);
    }

    static glm::mat4 composeLocal(const JointPose& pose) {
        return glm::translate(glm::mat4(1.0f), pose.translate) *
               glm::mat4_cast(pose.rotation) *
               glm::scale(glm::mat4(1.0f), pose.scale);
    }
};

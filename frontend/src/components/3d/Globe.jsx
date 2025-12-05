import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Sphere, MeshDistortMaterial } from '@react-three/drei';

export default function Globe() {
    const sphereRef = useRef();

    useFrame((state) => {
        const t = state.clock.getElapsedTime();
        if (sphereRef.current) {
            sphereRef.current.rotation.y = t * 0.2;
            sphereRef.current.rotation.z = t * 0.05;
        }
    });

    return (
        <group>
            <ambientLight intensity={0.5} />
            <directionalLight position={[10, 10, 5]} intensity={1} />
            <pointLight position={[-10, -10, -10]} color="#FF4B4B" intensity={2} />

            <Sphere ref={sphereRef} args={[1, 64, 64]} scale={2.5}>
                <MeshDistortMaterial
                    color="#1E293B"
                    attach="material"
                    distort={0.4}
                    speed={1.5}
                    roughness={0.2}
                    metalness={0.8}
                    wireframe={true}
                />
            </Sphere>

            {/* Core Glow */}
            <Sphere args={[0.8, 32, 32]} scale={2.4}>
                <meshBasicMaterial color="#FF4B4B" transparent opacity={0.1} />
            </Sphere>
        </group>
    );
}

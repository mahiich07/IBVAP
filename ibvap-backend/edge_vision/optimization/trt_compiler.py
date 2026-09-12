"""
Edge Model Optimization: TensorRT Engine Compiler & Quantization
Provides Python TensorRT API integration, INT8/FP16 quantization, DLA core scheduling, and trtexec scripts.
"""

import os
import subprocess
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger("ibvap.trt_compiler")


class TensorRTCompiler:
    """
    Compiles ONNX neural networks into optimized serialized NVIDIA TensorRT (.engine) binaries.
    Tailored for Jetson Orin Nano, AGX Orin, and Xavier NX edge processors.
    """

    def __init__(
        self,
        onnx_path: str,
        engine_path: Optional[str] = None,
        precision: str = "fp16",  # 'fp32', 'fp16', 'int8'
        workspace_mb: int = 4096,
        use_dla: bool = False,
        dla_core: int = 0,
        batch_size: int = 1
    ):
        self.onnx_path = onnx_path
        if not engine_path:
            base, _ = os.path.splitext(onnx_path)
            self.engine_path = f"{base}_{precision}.engine"
        else:
            self.engine_path = engine_path

        self.precision = precision.lower()
        self.workspace_mb = workspace_mb
        self.use_dla = use_dla
        self.dla_core = dla_core
        self.batch_size = batch_size

    def build_trtexec_command(self) -> str:
        """
        Constructs the optimized standalone trtexec shell command for compiling on NVIDIA Jetson.
        """
        cmd_parts = [
            "trtexec",
            f"--onnx={self.onnx_path}",
            f"--saveEngine={self.engine_path}",
            f"--memPoolSize=workspace:{self.workspace_mb}MiB"
        ]

        if self.precision == "fp16":
            cmd_parts.append("--fp16")
        elif self.precision == "int8":
            cmd_parts.extend(["--int8", "--calib=calibration.cache"])

        if self.use_dla:
            # Jetson Orin/Xavier deep learning accelerator cores
            cmd_parts.extend([
                f"--useDLACore={self.dla_core}",
                "--allowGPUFallback"
            ])

        # Dynamic shape specification for batch 1 to 4
        cmd_parts.extend([
            f"--minShapes=images:1x3x640x640",
            f"--optShapes=images:{self.batch_size}x3x640x640",
            f"--maxShapes=images:4x3x640x640"
        ])

        return " ".join(cmd_parts)

    def compile_with_python_api(self) -> bool:
        """
        Attempts to compile the engine directly using the native Python TensorRT API on Jetson.
        """
        try:
            import tensorrt as trt
            logger.info("Initializing TensorRT Python Builder...")

            TRT_LOGGER = trt.Logger(trt.Logger.INFO)
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, TRT_LOGGER)

            with open(self.onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for error in range(parser.num_errors):
                        logger.error(f"TRT Parser Error: {parser.get_error(error)}")
                    return False

            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.workspace_mb * (1024 * 1024))

            if self.precision == "fp16":
                if builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                    logger.info("FP16 Half-Precision mode enabled.")

            if self.precision == "int8":
                if builder.platform_has_fast_int8:
                    config.set_flag(trt.BuilderFlag.INT8)
                    logger.info("INT8 Quantization mode enabled.")

            if self.use_dla and builder.num_DLA_cores > 0:
                config.default_device_type = trt.DeviceType.DLA
                config.DLA_core = self.dla_core
                config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
                logger.info(f"Targeting Jetson DLA Core {self.dla_core} with GPU fallback.")

            logger.info(f"Building serialized TensorRT engine -> {self.engine_path}...")
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                logger.error("Failed to build TensorRT serialized network.")
                return False

            with open(self.engine_path, "wb") as f:
                f.write(serialized_engine)

            logger.info(f"TensorRT Engine successfully saved to {self.engine_path}")
            return True

        except ImportError:
            logger.info("NVIDIA TensorRT Python library not installed in host environment. Use trtexec CLI command.")
            return False
        except Exception as e:
            logger.error(f"TensorRT Python compilation failed: {e}")
            return False

    def generate_shell_script(self, output_sh: str = "compile_trt.sh") -> str:
        """Writes executable bash script for running trtexec directly on the edge board."""
        trtexec_cmd = self.build_trtexec_command()
        content = (
            "#!/bin/bash\n"
            "# IBVAP Edge TensorRT Engine Compiler Script for NVIDIA Jetson\n"
            "set -e\n\n"
            f"echo 'Compiling {self.onnx_path} to TensorRT {self.precision.upper()} engine...'\n"
            f"{trtexec_cmd}\n"
            "echo 'Compilation complete!'\n"
        )
        with open(output_sh, "w") as f:
            f.write(content)
        os.chmod(output_sh, 0o755)
        return output_sh

"""Load balancing module for Flow2API"""
import random
from typing import Optional
from ..core.models import Token
from .concurrency_manager import ConcurrencyManager
from ..core.logger import debug_logger


class LoadBalancer:
    """Token load balancer with random selection"""

    # 简化后的等级定义
    TIER_LEVELS = {
        "standard": 1,  # 普号 (1k/Flash)
        "pro": 2,       # 会员/Pro (2k) - 对应 PAYGATE_TIER_ONE
        "ultra": 3      # Ultra (4k) - 对应 PAYGATE_TIER_TWO
    }

    # 上游原始字段映射
    RAW_TIER_MAP = {
        None: "standard",
        "": "standard",
        "PAYGATE_TIER_ONE": "pro",
        "PAYGATE_TIER_TWO": "ultra"
    }

    def __init__(self, token_manager, concurrency_manager: Optional[ConcurrencyManager] = None):
        self.token_manager = token_manager
        self.concurrency_manager = concurrency_manager

    async def select_token(
        self,
        for_image_generation: bool = False,
        for_video_generation: bool = False,
        model: Optional[str] = None,
        required_tier: Optional[str] = None
    ) -> Optional[Token]:
        """
        Select a token using random load balancing

        Args:
            required_tier: "standard" | "pro" | "ultra"
        """
        debug_logger.log_info(f"[LOAD_BALANCER] 开始选择Token (图片={for_image_generation}, 视频={for_video_generation}, 模型={model}, Tier>={required_tier})")

        active_tokens = await self.token_manager.get_active_tokens()
        debug_logger.log_info(f"[LOAD_BALANCER] 获取到 {len(active_tokens)} 个活跃Token")

        if not active_tokens:
            debug_logger.log_info(f"[LOAD_BALANCER] ❌ 没有活跃的Token")
            return None

        # Filter tokens based on generation type
        available_tokens = []
        filtered_reasons = {}  # 记录过滤原因
        
        # 1. 确定所需的数字等级 (默认 standard=1)
        req_level = self.TIER_LEVELS.get(required_tier, 1) if required_tier else 1

        for token in active_tokens:
            # 2. 转换Token等级: 原始字符串 -> 简化名 -> 数字等级
            # 默认为 standard
            token_tier_name = self.RAW_TIER_MAP.get(token.user_paygate_tier, "standard")
            token_level = self.TIER_LEVELS.get(token_tier_name, 1)

            if token_level < req_level:
                filtered_reasons[token.id] = f"权限不足 (需 {required_tier or 'standard'}, 当前 {token_tier_name})"
                continue

            # Check if token has valid AT (not expired)
            if not await self.token_manager.is_at_valid(token.id):
                filtered_reasons[token.id] = "AT无效或已过期"
                continue

            # Filter for image generation
            if for_image_generation:
                if not token.image_enabled:
                    filtered_reasons[token.id] = "图片生成已禁用"
                    continue

                # Check concurrency limit
                if self.concurrency_manager and not await self.concurrency_manager.can_use_image(token.id):
                    filtered_reasons[token.id] = "图片并发已满"
                    continue

            # Filter for video generation
            if for_video_generation:
                if not token.video_enabled:
                    filtered_reasons[token.id] = "视频生成已禁用"
                    continue

                # Check concurrency limit
                if self.concurrency_manager and not await self.concurrency_manager.can_use_video(token.id):
                    filtered_reasons[token.id] = "视频并发已满"
                    continue

            available_tokens.append(token)

        # 输出过滤信息
        if filtered_reasons:
            debug_logger.log_info(f"[LOAD_BALANCER] 已过滤Token:")
            for token_id, reason in filtered_reasons.items():
                debug_logger.log_info(f"[LOAD_BALANCER]   - Token {token_id}: {reason}")

        if not available_tokens:
            debug_logger.log_info(f"[LOAD_BALANCER] ❌ 没有可用的Token (图片生成={for_image_generation}, 视频生成={for_video_generation})")
            return None

        # Random selection
        selected = random.choice(available_tokens)
        debug_logger.log_info(f"[LOAD_BALANCER] ✅ 已选择Token {selected.id} ({selected.email}) - 余额: {selected.credits}")
        return selected

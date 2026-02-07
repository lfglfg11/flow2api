# 开发日志 (2026-01-29)

## 1. Gemini API 接口重构与增强
### 1.1 认证机制升级
- **灵活认证**: 废弃原有的固定 header 验证，实现 `verify_gemini_auth`。
    - 支持 `x-goog-api-key` Header。
    - 支持 `key` 查询参数 (Query Param)。
    - 支持 `Authorization: Bearer <token>`。
- **兼容性**: 完美适配 Gemini 官方 SDK 及第三方客户端的调用习惯。

### 1.2 多模态输入解析增强
- **URL 图片支持**: 修复了 `parse_gemini_content` 中仅支持 base64 的缺陷。
    - 新增逻辑：当 `inline_data` 的 data 字段为 HTTP/HTTPS 链接时，自动调用 `retrieve_image_data` 下载图片。
- **Base64 容错增强**:
    - 自动清洗 base64 字符串中的换行符 (`\n`, `\r`) 和空格。
    - 智能识别并去除 `base64,` 前缀。
    - 自动补全缺失的 padding (`=`)，解决解码失败问题。

### 1.3 错误处理优化
- **上游错误透传**: 在处理 Gemini 响应时，增加对 `error` 字段的检查。
    - 遇到生成失败（如 400/500）时，不再返回空文本，而是抛出包含具体错误信息的 HTTP 500 异常，便于客户端定位问题。

## 2. 图像上传服务 (Flow Client) 强化
### 2.1 格式标准化
- **强制 JPEG 转码**: 解决 Flow 上游接口对图片格式（某些 WebP/PNG/Base64）挑剔导致的 `400 Invalid Argument` 错误。
    - 引入 `Pillow` 库，在上传前将所有图片强制转换为标准 JPEG (RGB模式, Quality 95)。
    - 强制设定 MIME 类型为 `image/jpeg`。
- **性能平滑**:
    - **线程池卸载**: 将 CPU 密集的图片转码操作 (`_convert_to_jpeg`) 移至 `asyncio` 线程池 (`run_in_executor`) 执行，确保在高并发下**不阻塞**主事件循环。

## 3. Token 分级与号池管理
### 3.1 模型分级配置
- **配置更新**: 在 `MODEL_CONFIG` 中引入 `required_tier` 字段。
    - `PAYGATE_TIER_THREE` (Ultra): 适用于所有 4K 模型 (`*-4k`) 及 Ultra 系列模型。
    - `PAYGATE_TIER_TWO` (Pro): 适用于 2K 模型 (`*-2k`)。
    - `Default` (Ordinary): 默认 Tier 1，适用于普通 1K 模型。

### 3.2 智能负载均衡
- **权限路由**: 更新 `LoadBalancer.select_token` 逻辑。
    - 引入 `TIER_LEVELS` 映射 (One->1, Two->2, Three->3)。
    - 实现了**前置极速过滤**：在查库前优先检查 `token.user_paygate_tier` 是否满足模型要求的 `required_tier`。
    - **向下兼容**: 高等级 Token 可用于低等级请求，反之则过滤。

---
*注：以上修改均已通过测试（curl/python client），并针对高并发场景进行了非阻塞优化。*

# 开发日志 (2026-02-08)

## 1. Gemini 接口返回格式对齐
### 1.1 非流式响应标准化
- **补齐核心字段**: 增加 `usageMetadata`、`createTime`、`responseId`、`id`、`object`、`created`、`choices`、`usage`。
- **Token 统计映射**: 从上游 OpenAI `usage` 映射到 Gemini `usageMetadata` 与 `usage`。

### 1.2 流式响应标准化
- **SSE 数据结构对齐**: 流式输出使用 Gemini `candidates` 结构，结束包附带 `finishReason=STOP`。
- **结束信号修正**: 遇到上游 `[DONE]` 时，转换为 Gemini 结束包，避免非标准结束标记。

### 1.3 请求参数兼容
- **generationConfig 支持**: 解析 `generationConfig` 与 `generation_config` 两种写法，确保与官方文档一致。

## 2. Gemini 图片返回内联修正
### 2.1 图片 parts 结构
- **inline_data 对齐**: 将生成图片放入 `candidates.content.parts.inline_data`，避免落在外层文本或 `choices` 中。
- **图片提取与编码**: 识别 Markdown 图片链接或 data URI，必要时下载并转为 base64 以符合 Gemini 规范。

### 2.2 流式结束包修正
- **标准结束包**: 在流式结束时补发携带 `inline_data` 的结束包（`finishReason=STOP`）。

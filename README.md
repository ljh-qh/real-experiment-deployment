# Real Experiment Deployment

OpenWAM / EasyWAM 双臂与 Sharpa 手部署代码快照。

主要代码位于 deployment/。最新配置见 [部署配置总结](deployment/DEPLOYMENT_CONFIG_SUMMARY.md)。
deployment/README.md 包含迁移历史，其中较早的模型数量与 RTC 描述以配置总结为准。

## 不包含的内容

模型权重、基础模型、归一化张量及文本 context 张量、相机参考图像、示范数据、实验日志、已安装虚拟环境及第三方原生 SDK 均不上传。
权重目录保留 .gitkeep；assets/checkpoints/models.json 保留文件名与 SHA256。
.gitignore 同时防止常见权重和数据格式被误提交。

这不是下载后即可直接驱动机器人的完整运行包。恢复部署前需要：
1. 从有权限的来源恢复对应 checkpoint、stats、context、tokenizer 与 Cosmos 基础模型，按登记哈希核验。
2. 获取并安装 Tianji / Sharpa SDK，恢复项目内 envs 与 deps。版本清单见 requirements/，它们是快照而非已验证的通用安装锁文件。
3. 恢复各任务参考图像、state80 与必要测试数据。已有 pytest/unittest 部分测试依赖这些未上传资源。
4. 检查机器人地址、相机序列号、SDK路径、URDF、起点与硬件限位；当前配置包含原实验室的路径和设备标识。
5. 先执行离线验证，重新生成相应 validation 文件。不能把原机器实验通过视作新设备已验证。

不附带历史真实运行的 hardware_ready 授权。源码导出中的验证入口保持失败关闭。
统一入口为 deployment/scripts/launch.py；未完成资源恢复前路径检查预期会失败。

## 来源与许可

vendor/ 为当前部署使用的第三方代码快照，保留已有 LICENSE、NOTICE 和 README。
本仓库不额外授予第三方代码或硬件 SDK 的许可；使用和再分发遵循对应来源条款。
发布快照不会改写服务器上正在使用的 deployment 文件。

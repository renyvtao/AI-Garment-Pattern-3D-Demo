# Windows 访问远程服务

## 配置连接

复制连接模板：

```text
deployment/server_connection.example.json
```

保存为不进入 Git 的 `deployment/server_connection.local.json`，或者运行：

```powershell
./deployment/update_server.ps1
```

填写 SSH 主机、端口、用户名、远程项目路径和本地端口。配置文件不接收或保存密码。

## 启动

双击 `deployment/open_demo.cmd`，或在 PowerShell 中执行：

```powershell
./deployment/connect_server.ps1
```

脚本会完成以下工作：

1. 打开独立的 SSH 窗口；
2. 提示输入一次 SSH 密码（输入时不会显示字符）；
3. 在服务器上启动人体服务、任务服务、网页服务和统一网关；
4. 建立 `http://127.0.0.1:3000` 到服务器的安全隧道；
5. 等待网页和两个 API 健康检查通过；
6. 自动用默认浏览器打开结果网页。

SSH 窗口出现 `AI_GARMENT_READY` 后表示服务器端服务已经就绪。使用网页期间必须保持该窗口开启；关闭窗口会断开本机访问通道，但不会停止服务器端后台服务。

程序不会保存 SSH 密码。首次连接某个新实例时，还会要求确认主机指纹；核对无误后输入 `yes`，再输入密码。

## 修改连接信息

运行：

```text
deployment\update_server.cmd
```

按照提示填写新的 SSH 主机和端口，其余项目按回车保留。配置文件不保存密码。

也可在 PowerShell 中执行：

```powershell
./deployment/update_server.ps1 `
  -HostName gpu.example.com `
  -Port 22 `
  -ProfileName 'GPU Server' `
  -RemoteProjectRoot '/opt/ai-garment-pattern-3d-demo'
```

本地端口需要修改时传入：

```powershell
-LocalPort 3002
```

`127.0.0.1:<localPort>` 是当前电脑上的 SSH 隧道入口。模型推理、仿真和产物文件仍位于远程服务器。

## 网页功能

- 单张或多张图片上传，批量任务共用一套人体尺寸；
- 女性、男性、中性预设人体或定制 SMPL-X 人体；
- 原始展示动作等当前已部署的动作选项；
- 显示任务排队、推理、缝合、仿真、收集产物等状态；
- 预览输入图、板片、静态渲染和视频；
- 按需展开体型、基础人体 ID、26 字段 YAML、模型原始文本和规格文件；
- 单文件下载、整任务 ZIP 下载以及用户控制的任务删除。

## 安全说明

- 不要把 SSH 密码写入脚本、JSON 或发送给其他人；
- 程序保留 SSH 主机指纹校验，不会静默接受被替换的服务器；
- SSH 隧道适合开发和单机访问。多人服务应部署反向代理、HTTPS、身份认证和访问控制。

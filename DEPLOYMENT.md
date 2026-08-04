# 部署指南（Deployment Guide）

本项目的本地环境（WSL + Windows 盘）做了特殊处理，**部署到服务器时不需要这些处理**，服务器上是标准流程。本文件说明：本地怎么改/编译、上传什么、服务器怎么部署。

## 一、本地开发（WSL 环境）

本项目跑在 `/mnt/e`（WSL 挂载的 Windows 盘），`node_modules` 是指向 ext4 的符号链接（`/root/fuwari-node_modules`），用于绕开 WSL 9P 协议卡死问题。

| 操作 | 命令 | 说明 |
|---|---|---|
| 启动 dev | `corepack pnpm dev` | 不要用 `npm run dev` |
| 构建 | `corepack pnpm build` | 输出到 `dist/`，构建后会跑 pagefind |
| 装/更新依赖 | `bash scripts/pnpm-install.sh` | 直接 `pnpm install` 会因符号链接报 ENOTDIR |
| 修改 `src/config.ts` | — | **改完需手动重启 dev**（Astro 不监听该文件） |

注意：
- 编辑 `src/` 下的代码用 Windows 侧编辑器（VS Code）或 WSL 侧都行，dev 服务器会自动热更新。
- `astro.config.mjs` 里的 `vite.server.fs.allow` **只影响 dev 服务器**，生产构建不需要。

## 二、上传到服务器（关键）

**只上传源码，不要上传 `node_modules`。**

`node_modules` 在 Windows 侧是一个指向 WSL 目录的符号链接，直接上传会损坏或失效；服务器上需要重新安装。

✅ **需要上传**（建议通过 git 仓库同步，最干净）：
```
src/            # 全部源码
public/         # 静态资源
package.json    # 含 @swup/astro 的 peer 修复（pnpm.packageExtensions）
pnpm-lock.yaml  # 锁定依赖版本
astro.config.mjs
tsconfig.json
svelte.config.js
tailwind.config.cjs
postcss.config.mjs
biome.json
frontmatter.json
pagefind.yml
vercel.json（如用 Vercel）
```

❌ **不要上传**：
```
node_modules/   # 符号链接，服务器上重新 pnpm install
dist/           # 构建产物，服务器上重新 build
.astro/         # 构建缓存
.git/           # （如用 git 部署则自动排除）
```

## 三、服务器部署

### 方式 A：自建服务器（Nginx / Caddy 等）

```bash
# 1. 环境要求：Node.js 20+（推荐 22 LTS）、pnpm
curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && apt install -y nodejs
corepack enable

# 2. 上传源码到服务器后，进入项目目录
corepack pnpm install --frozen-lockfile   # 标准安装，服务器上无符号链接问题
corepack pnpm build                       # 产出 dist/

# 3. 用 Nginx 托管 dist/，示例配置：
#    server {
#        listen 80;
#        server_name your-domain.com;
#        root /var/www/fuwari/dist;
#        location / { try_files $uri $uri.html $uri/ =404; }
#    }
```

### 方式 B：托管平台（Vercel / Netlify / Cloudflare Pages）

- 直接用 git 仓库连接，平台自动识别 Astro。
- **无需上传任何文件**，推送代码即可，平台自动 `pnpm install && pnpm build`。
- `astro.config.mjs` 里的 `site` 字段记得改成你的正式域名。
- `vercel.json` 已存在，Vercel 可零配置部署。

### 方式 C：Cloudflare Pages（推荐）

本项目是纯静态站，Cloudflare Pages 足够（**不需要 Workers**）。

**1. 推送到 GitHub 后，Git 集成自动部署（推荐）：**

1. Cloudflare 控制台 → **Workers & Pages** → **Create** → **Pages** → **Connect to Git** → 授权 GitHub → 选择仓库。
2. 构建设置：
   - Framework preset：**Astro**
   - Build command：`pnpm run build`（框架预设已自动填 `astro build`，改成含 pagefind 的完整命令）
   - Build output directory：`dist`
   - 环境变量：新增 `NODE_VERSION=22`（显式指定 Node 版本，避免默认版本过旧）
3. 点击 **Save and Deploy**，等待构建完成，访问 `https://<项目名>.pages.dev`。
4. 以后 `git push` 自动触发重新构建部署。

**2. 或使用 Wrangler CLI（手动上传 dist）：**

```bash
npm install -g wrangler        # 一次性安装
pnpm run build                 # 本地构建出 dist/
npx wrangler pages project create fuwari-blog
npx wrangler pages deploy dist --project-name fuwari-blog
```

**3. 自定义域名（可选）：**

Pages → 项目 → **Custom domains** → Add custom domain。域名 DNS 已托管在 Cloudflare 的话可一键接入。

**4. 部署后检查：**

- ✅ `site` 字段：`astro.config.mjs` 里改成你的正式域名（影响 RSS / sitemap 链接）
- ✅ 站内搜索（pagefind）：构建命令已含 `pagefind --site dist`，CF 构建环境可正常运行
- ✅ `vercel.json` 对 CF 无影响，可忽略

## 四、部署前的自检清单

1. ✅ 停掉本地 dev 服务器后运行 `corepack pnpm build`，确认 `dist/` 非空。
2. ✅ 检查 `src/config.ts`：`siteConfig.title`、`site` 域名、`banner.src`、`profileConfig` 等是否已改成你的信息。
3. ✅ `package.json` 的 `pnpm.packageExtensions`（@swup/astro 修复）随代码一起提交。
4. ⚠️ 仓库文件是 CRLF 换行（Windows checkout 产生）。JS/TS/HTML/MD 不受影响；**若有 `.sh` 脚本需在服务器上执行，先转 LF**（`sed -i 's/\r$//' script.sh`）。本项目唯一脚本 `scripts/pnpm-install.sh` 是 WSL 专用，无需部署到服务器。

## 五、已知问题（不影响部署）

- **Svelte 水合错误**：`Search.svelte` / `LightDarkSwitch.svelte` / `DisplaySettings.svelte` / `DirectoryTreeNode.svelte` 在 dev 下有 `effect_orphan` / `Cannot read properties of null` 报错（控制台可见），导致搜索、主题切换、目录树交互异常。属于既有代码问题（与部署无关），修复方式：排查这些组件的 `$effect`/`onMount` 用法与 SSR 结构一致性。

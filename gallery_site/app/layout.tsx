import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI 服装制版与三维展示",
  description: "从服装图片生成二维板片、静态三维服装与动态布料视频。",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}

import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { headers } from "next/headers";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("x-forwarded-host") ?? requestHeaders.get("host");
  const protocol =
    requestHeaders.get("x-forwarded-proto") ??
    (host?.startsWith("localhost") ? "http" : "https");
  const metadataBase = new URL(`${protocol}://${host ?? "localhost:3000"}`);

  return {
    metadataBase,
    title: "Capsule · 个人多模态素材工作台",
    description:
      "导入、处理、浏览、聚类与检索个人多模态素材，并把发现保存成可回放的 Capsule。",
    icons: {
      icon: "/favicon.svg",
      shortcut: "/favicon.svg",
    },
    openGraph: {
      title: "Capsule · 个人多模态素材工作台",
      description: "从散落素材到可检索、可聚类、可回放的个人记忆库。",
      images: [
        {
          url: new URL("/og-workspace.png", metadataBase).toString(),
          width: 1761,
          height: 893,
          alt: "Capsule 个人多模态素材工作台",
        },
      ],
    },
    twitter: {
      card: "summary_large_image",
      title: "Capsule · 个人多模态素材工作台",
      description: "从散落素材到可检索、可聚类、可回放的个人记忆库。",
      images: [new URL("/og-workspace.png", metadataBase).toString()],
    },
  };
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}

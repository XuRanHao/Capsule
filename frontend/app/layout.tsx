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
    title: "Capsule · 多模态记忆检索",
    description:
      "用文字、图片或图文组合，从个人素材库中检索 Asset、来源文件与关联段落。",
    icons: {
      icon: "/favicon.svg",
      shortcut: "/favicon.svg",
    },
    openGraph: {
      title: "Capsule · 多模态记忆检索",
      description: "搜到你记得的那一幕。",
      images: [
        {
          url: new URL("/og.png", metadataBase).toString(),
          width: 1792,
          height: 909,
          alt: "Capsule 多模态记忆检索",
        },
      ],
    },
    twitter: {
      card: "summary_large_image",
      title: "Capsule · 多模态记忆检索",
      description: "搜到你记得的那一幕。",
      images: [new URL("/og.png", metadataBase).toString()],
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

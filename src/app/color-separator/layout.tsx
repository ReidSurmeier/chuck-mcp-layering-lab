import { unstable_noStore as noStore } from "next/cache";
export const revalidate = 0;
export const dynamic = "force-dynamic";
export const fetchCache = "force-no-store";

import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "color.separator — digital color separation tool",
  description: "Digital color separation for woodblock, CNC, and silkscreen printing",
};

export default function ColorSeparatorLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  noStore(); // Disable Next.js Full Route Cache
  return <>{children}</>;
}

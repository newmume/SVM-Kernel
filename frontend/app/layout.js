import "./globals.css";

export const metadata = {
  title: "SVM Kernel Lab · V2",
  description: "Interactive SVM kernel and decision-function explorer",
};

export default function RootLayout({ children }) {
  return (
    <html lang="zh-Hant">
      <body>{children}</body>
    </html>
  );
}

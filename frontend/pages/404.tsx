import React from 'react';
import Link from 'next/link';
import Head from 'next/head';
import Header from '../components/Header';

const NotFoundPage: React.FC = () => {
  return (
    <div className="min-h-screen flex flex-col">
      <Head>
        <title>页面未找到 - 实时语音转写系统</title>
        <meta name="description" content="404 - 页面未找到" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <Header />

      <main className="container mx-auto px-4 py-12 flex-grow flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-6xl font-bold text-primary-500 mb-4">404</h1>
          <h2 className="text-2xl font-semibold text-gray-700 dark:text-gray-300 mb-6">页面未找到</h2>
          <p className="text-gray-600 dark:text-gray-400 mb-8">
            您访问的页面不存在或已被移动
          </p>
          <Link href="/">
            <span className="px-6 py-3 bg-primary-500 text-white rounded-md hover:bg-primary-600 transition-colors">
              返回首页
            </span>
          </Link>
        </div>
      </main>

      <footer className="py-6 border-t border-gray-200 dark:border-gray-800">
        <div className="container mx-auto px-4 text-center text-gray-500 dark:text-gray-400">
          <p>© {new Date().getFullYear()} 实时语音转写系统 | 基于FastAPI和Next.js开发</p>
        </div>
      </footer>
    </div>
  );
};

export default NotFoundPage; 
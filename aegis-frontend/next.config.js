/** @type {import('next').NextConfig} */
const nextConfig = {
  // Rewrite /api/* calls to the FastAPI backend during development
  // This avoids browser-level CORS issues by proxying through Next.js server
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${process.env.AEGIS_API_URL || 'http://127.0.0.1:8000'}/api/:path*`,
      },
      {
        source: '/ws/:path*',
        destination: `${process.env.AEGIS_API_URL || 'http://127.0.0.1:8000'}/ws/:path*`,
      },
    ];
  },

  // Allow images from any source (for avatars, charts, etc.)
  images: {
    remotePatterns: [
      { protocol: 'http', hostname: 'localhost' },
      { protocol: 'https', hostname: '**' },
    ],
  },
};

module.exports = nextConfig;

/** @type {import('next-sitemap').IConfig} */
const config = {
  siteUrl: "https://yourdomain.com", // TODO: Add your domain
  generateRobotsTxt: true,
  exclude: ["/twitter-image.*", "/opengraph-image.*", "/icon.*"],
};

export default config;

const path = require('path');

module.exports = {
  entry: './main.js',
  output: {
    filename: 'bundle.js',
    path: path.resolve(__dirname, 'dist'),
  },
  mode: 'development',
  experiments: {
    asyncWebAssembly: true,
  },
  module: {
    rules: [
      {
        test: /\.worker\.js$/,
        use: { 
          loader: 'worker-loader',
          options: { 
            inline: 'no-fallback'
          }
        }
      }
    ]
  }
};

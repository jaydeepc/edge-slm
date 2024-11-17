import CopyWebpackPlugin from 'copy-webpack-plugin';
import { fileURLToPath } from 'url';
import path from 'path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default {
    mode: 'development',
    devtool: 'source-map',
    entry: {
        'dist/main': './main.js',
        'dist/main.min': './main.js',
    },
    output: {
        filename: '[name].js',
        path: __dirname,
        library: {
            type: 'module',
        },
    },
    module: {
        rules: [
            {
                test: /\.svg$/,
                use: 'raw-loader'
            },
            {
                test: /\.js$/,
                resourceQuery: /raw/,
                type: 'asset/source'
            }
        ]
    },
    plugins: [
        new CopyWebpackPlugin({
            patterns: [
                {
                    from: 'node_modules/onnxruntime-web/dist/*.jsep.*',
                    to: 'dist/[name][ext]'
                }
            ],
        }),
    ],
    devServer: {
        static: {
            directory: __dirname
        },
        port: 8080,
        hot: false,
        liveReload: true
    },
    experiments: {
        outputModule: true,
    },
    resolve: {
        fallback: {
            "fs": false,
            "path": false,
            "url": false
        }
    }
};

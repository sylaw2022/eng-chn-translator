# Deploying to Vercel

To deploy this static UI to Vercel:

1.  **Install Vercel CLI**:
    ```bash
    npm install -g vercel
    ```

2.  **Login to Vercel**:
    ```bash
    vercel login
    ```
    *Follow the instructions in your terminal/browser.*

3.  **Deploy**:
    ```bash
    vercel
    ```
    *   Accept default settings (just press Enter).
    *   Your site will be deployed to a URL like `https://eng-chn-translator-ui.vercel.app`.

## Important Note on Backend Connection

Currently, `index.html` is configured to connect to `http://localhost:9000/translate`.

**This will NOT work** on the deployed Vercel site because Vercel (public internet) cannot access your `localhost`.

To make it work:
1.  **Deploy your Backend** (the `app.py` / Docker container) to a public cloud provider like Google Cloud Run (see `DEPLOY_GCP.md`).
2.  **Get the Public URL** of your backend (e.g., `https://translator-app-xyz.a.run.app`).
3.  **Update `index.html`**:
    *   Find line `const API_URL = 'http://localhost:9000/translate';`
    *   Replace it with your new public URL: `const API_URL = 'https://translator-app-xyz.a.run.app/translate';`
4.  **Redeploy to Vercel**:
    ```bash
    vercel --prod
    ```


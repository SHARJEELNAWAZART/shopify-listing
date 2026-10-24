import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3000;

// Increase JSON and urlencoded body limits for high-resolution furniture image uploads
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

const SYSTEM_PROMPT = `You are an expert furniture copywriter and technical Shopify SEO specialist. 
Your task is to analyze the provided furniture image and user specifications, then output a complete, conversion-optimized Shopify listing.

CRITICAL GUARDRAILS (YOU MUST FOLLOW THESE STRICTLY):
1. THE PRIMARY SUBJECT ANCHOR RULE:
   Focus solely on the item explicitly named in the "Target Product Name". If the image is a staged lifestyle scene containing secondary furniture or decor (e.g., rugs, lamps, chairs around a table), treat them strictly as staging accessories. If secondary items are visible, append this exact disclaimer at the end of the HTML description: "<p><em>Note: Staging accessories and additional furniture pictured are sold separately.</em></p>".

2. THE ZERO-ASSUMPTION SPEC RULE:
   Never invent or assume internal construction details, joinery methods, cushion core fills (e.g., do not guess 'high-resilience foam' or 'down feather'), weight limits, or wood grades. If a spec is not explicitly mentioned in the user's text, omit it or describe the piece using observable visual language only (e.g., "generously padded seating", "slender tapered legs").

3. THE TEXT-FIRST COLOR PRECEDENCE RULE:
   The user's text input is the absolute truth for colors and finishes. Lighting and filters in images can distort colors (e.g., making cognac look espresso, or green look slate). If the user provides a specific colorway name, use that exact name. If no colorway is provided, describe the color tone generally and add "[Verify Color]" in the tags.

FORMATTING REQUIREMENTS:
- body_html: Output semantic HTML without inline styles. Use an evocative 2-paragraph lifestyle hook, an <h3>Features & Details</h3> section with a <ul> list, and an <h3>Dimensions & Care</h3> section.
- meta_title: Under 60 characters. Include primary product type and key aesthetic.
- meta_description: 145 to 160 characters. Compelling summary ending with a call to action (e.g., "Shop now.").
- image_alt_text: Under 125 characters. Highly descriptive of the subject, material, and color for accessibility and search crawlers.
- tags: Array of 5-8 relevant comma-separated tags (style, material, room, category).

RESPONSE FORMAT:
You MUST respond with a valid, raw JSON object matching this schema exactly (no markdown formatting, no triple backticks):
{
  "product_title_suggestion": "string",
  "body_html": "string",
  "meta_title": "string",
  "meta_description": "string",
  "image_alt_text": "string",
  "tags": ["string", "string"]
}`;

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    hasGeminiKey: Boolean(process.env.GEMINI_API_KEY),
    time: new Date().toISOString()
  });
});

// Helper to parse base64 data URLs
function extractBase64Data(dataUrl: string): { mimeType: string; base64: string } {
  if (!dataUrl) {
    return { mimeType: 'image/jpeg', base64: '' };
  }
  const match = dataUrl.match(/^data:([^;]+);base64,(.+)$/);
  if (match) {
    return { mimeType: match[1], base64: match[2] };
  }
  // If raw base64 or SVG
  if (dataUrl.startsWith('data:')) {
    const commaIndex = dataUrl.indexOf(',');
    if (commaIndex !== -1) {
      const mime = dataUrl.substring(5, commaIndex).split(';')[0] || 'image/jpeg';
      const raw = dataUrl.substring(commaIndex + 1);
      return { mimeType: mime, base64: Buffer.from(decodeURIComponent(raw)).toString('base64') };
    }
  }
  return { mimeType: 'image/jpeg', base64: dataUrl };
}

// POST /api/generate
app.post('/api/generate', async (req, res) => {
  try {
    const { productName, specs, tone, imageBase64, customApiKey, provider } = req.body;

    if (!productName || !productName.trim()) {
      return res.status(400).json({ error: 'Missing Target Product Name' });
    }
    if (!specs || !specs.trim()) {
      return res.status(400).json({ error: 'Missing Key Specifications' });
    }
    if (!imageBase64) {
      return res.status(400).json({ error: 'Missing Product Image' });
    }

    const promptText = `Target Product Name: ${productName.trim()}
Brand Tone: ${tone || 'Warm & Editorial'}

Key Specifications & Details:
${specs.trim()}`;

    const { mimeType, base64 } = extractBase64Data(imageBase64);

    // Check if user requested OpenAI with custom key
    if (provider === 'openai' && customApiKey) {
      const openAiRes = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${customApiKey}`
        },
        body: JSON.stringify({
          model: 'gpt-4o-mini',
          response_format: { type: 'json_object' },
          messages: [
            { role: 'system', content: SYSTEM_PROMPT },
            {
              role: 'user',
              content: [
                { type: 'text', text: promptText },
                {
                  type: 'image_url',
                  image_url: { url: imageBase64.startsWith('data:') ? imageBase64 : `data:${mimeType};base64,${base64}` }
                }
              ]
            }
          ],
          temperature: 0.4
        })
      });

      if (!openAiRes.ok) {
        const errJson = await openAiRes.json().catch(() => ({}));
        const message = errJson.error?.message || `OpenAI returned status ${openAiRes.status}`;
        return res.status(openAiRes.status).json({ error: message });
      }

      const openAiData = await openAiRes.json();
      const content = openAiData.choices?.[0]?.message?.content;
      const parsed = JSON.parse(content);
      return res.json(parsed);
    }

    // Default: Use Gemini
    const apiKey = customApiKey || process.env.GEMINI_API_KEY;
    if (!apiKey) {
      return res.status(400).json({
        error: 'No Gemini API key available on the server or in request. Please check secrets or enter a key.'
      });
    }

    const ai = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build'
        }
      }
    });

    // Ensure mimeType is supported by Gemini Vision
    const supportedMimes = ['image/jpeg', 'image/png', 'image/webp'];
    const validMime = supportedMimes.includes(mimeType) ? mimeType : 'image/jpeg';

    const candidateModels = ['gemini-3.8-flash', 'gemini-3.1-flash-lite', 'gemini-flash-latest'];
    let lastError: any = null;
    let response: any = null;

    for (const modelName of candidateModels) {
      try {
        const contents: any[] = [];
        if (base64 && base64.length > 20) {
          contents.push({
            inlineData: {
              mimeType: validMime,
              data: base64
            }
          });
        }
        contents.push(promptText);

        response = await ai.models.generateContent({
          model: modelName,
          contents,
          config: {
            systemInstruction: SYSTEM_PROMPT,
            responseMimeType: 'application/json',
            temperature: 0.4,
            ...(modelName === 'gemini-3.8-flash' ? { thinkingConfig: { thinkingLevel: ThinkingLevel.MINIMAL } } : {}),
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                product_title_suggestion: { type: Type.STRING },
                body_html: { type: Type.STRING },
                meta_title: { type: Type.STRING },
                meta_description: { type: Type.STRING },
                image_alt_text: { type: Type.STRING },
                tags: {
                  type: Type.ARRAY,
                  items: { type: Type.STRING }
                }
              },
              required: [
                'product_title_suggestion',
                'body_html',
                'meta_title',
                'meta_description',
                'image_alt_text',
                'tags'
              ]
            }
          }
        });
        if (response && response.text) {
          break;
        }
      } catch (err: any) {
        lastError = err;
        console.warn(`Model ${modelName} encountered error:`, err.message || err);
        // Continue to next candidate model
      }
    }

    if (!response || !response.text) {
      throw lastError || new Error('All AI models were temporarily unavailable. Please retry in a moment.');
    }

    const rawText = response.text || '';
    let parsedData: any;
    try {
      let cleaned = rawText.trim();
      if (cleaned.startsWith('```')) {
        cleaned = cleaned.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '');
      }
      parsedData = JSON.parse(cleaned);
    } catch (parseErr) {
      console.warn('JSON parse issue, searching outer bounds:', rawText);
      const start = rawText.indexOf('{');
      const end = rawText.lastIndexOf('}');
      if (start !== -1 && end > start) {
        parsedData = JSON.parse(rawText.substring(start, end + 1));
      } else {
        throw new Error('AI response was not valid JSON');
      }
    }

    return res.json(parsedData);
  } catch (error: any) {
    console.error('API /api/generate Error:', error);
    const message = error.message || 'Server error while generating listing';
    return res.status(500).json({ error: message });
  }
});

// Mount Vite or static dist
async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true, hmr: false },
      appType: 'spa'
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist/index.html'));
    });
  }

  app.listen(Number(PORT), '0.0.0.0', () => {
    console.log(`Server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();

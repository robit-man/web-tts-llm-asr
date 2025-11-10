/// <reference types="vite/client" />

declare module "phonemizer" {
  export function phonemize(
    text: string,
    voice?: string,
  ):
    | string
    | string[]
    | { text?: string; phonemes?: string }
    | Promise<string | string[] | { text?: string; phonemes?: string }>;
}

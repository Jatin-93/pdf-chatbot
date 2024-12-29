import { NextResponse } from 'next/server'
import { Pinecone } from '@pinecone-database/pinecone'
import { OpenAIEmbeddings } from '@langchain/openai'
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter'
import fs from 'fs'
import path from 'path'
import pdf from 'pdf-parse'

let isInitialized = false

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY!
})

// Configure the PDF path using environment variable with fallback
const PDF_PATH = process.env.PDF_FILE_PATH || path.join(process.cwd(), 'public', 'book.pdf')

// Debug function to check paths
function debugPaths() {
  console.log('Current working directory:', process.cwd())
  console.log('Attempting to read PDF from:', PDF_PATH)
  console.log('Directory contents:', fs.readdirSync(process.cwd()))
  if (fs.existsSync('public')) {
    console.log('Public directory contents:', fs.readdirSync('public'))
  }
}

async function processPDF() {
  try {
    // Log debug information
    debugPaths()
    
    // Verify file exists
    if (!fs.existsSync(PDF_PATH)) {
      console.error('PDF file not found at:', PDF_PATH)
      throw new Error(`PDF file not found at path: ${PDF_PATH}`)
    }
    
    console.log('Reading PDF file...')
    const dataBuffer = fs.readFileSync(PDF_PATH)
    
    console.log('Parsing PDF content...')
    const data = await pdf(dataBuffer)
    
    if (!data || !data.text) {
      throw new Error('Failed to extract text from PDF')
    }
    
    console.log('Successfully extracted text from PDF')
    return data.text
  } catch (error: unknown) {
    console.error('PDF processing error:', error)
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred'
    throw new Error(`Failed to process PDF: ${errorMessage}`)
  }
}

async function initializePinecone() {
  if (isInitialized) return

  try {
    const index = pinecone.Index(process.env.PINECONE_INDEX_NAME!)
    
    console.log('Starting PDF processing...')
    const pdfText = await processPDF()
    
    console.log('Splitting text into chunks...')
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    })
    
    const chunks = await splitter.createDocuments([pdfText])
    console.log(`Created ${chunks.length} chunks`)

    const embeddings = new OpenAIEmbeddings({
      openAIApiKey: process.env.OPENAI_API_KEY,
    })
    
    console.log('Creating embeddings...')
    for (let i = 0; i < chunks.length; i += 100) {
      const batchChunks = chunks.slice(i, i + 100)
      const batchVectors = await Promise.all(
        batchChunks.map(async (chunk, index) => {
          const embedding = await embeddings.embedQuery(chunk.pageContent)
          return {
            id: `chunk_${i + index}`,
            values: embedding,
            metadata: { text: chunk.pageContent }
          }
        })
      )
      
      await index.upsert(batchVectors)
      console.log(`Uploaded batch ${i/100 + 1}/${Math.ceil(chunks.length/100)}`)
    }

    isInitialized = true
    console.log('Initialization complete')
  } catch (error: unknown) {
    console.error('Initialization error:', error)
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred'
    throw new Error(`Initialization failed: ${errorMessage}`)
  }
}

export async function POST(req: Request) {
  try {
    await initializePinecone()
    const index = pinecone.Index(process.env.PINECONE_INDEX_NAME!)

    const { query } = await req.json()
    const embeddings = new OpenAIEmbeddings({
      openAIApiKey: process.env.OPENAI_API_KEY,
    })
    
    const queryEmbedding = await embeddings.embedQuery(query)
    const queryResponse = await index.query({
      vector: queryEmbedding,
      topK: 5,
      includeMetadata: true
    })

    const context = queryResponse.matches?.map(match => match.metadata?.text).join('\n')
    
    const completion = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: 'gpt-3.5-turbo',
        messages: [
          {
            role: 'system',
            content: `You are a helpful assistant answering questions about a book. 
                     Use the following context to answer the question. 
                     If you're not sure about something, say so.
                     Context: ${context}`
          },
          {
            role: 'user',
            content: query
          }
        ],
        temperature: 0.7,
        max_tokens: 500
      })
    }).then(res => res.json())

    return NextResponse.json({
      answer: completion.choices[0].message.content
    })
  } catch (error: unknown) {
    console.error('Error in POST handler:', error)
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred'
    return NextResponse.json({ 
      error: `Failed to process request: ${errorMessage}`
    }, { status: 500 })
  }
}
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

// Updated PDF processing function with better error handling
async function processPDF() {
  try {
    // Adjust this path to match your PDF location
    const pdfPath = path.join(process.cwd(), 'public', 'book.pdf')
    
    // Check if file exists
    if (!fs.existsSync(pdfPath)) {
      throw new Error(`PDF file not found at path: ${pdfPath}`)
    }
    
    const dataBuffer = fs.readFileSync(pdfPath)
    const data = await pdf(dataBuffer)
    
    if (!data || !data.text) {
      throw new Error('Failed to extract text from PDF')
    }
    
    return data.text
  } catch (error) {
    console.error('PDF processing error:', error)
    throw new Error(`Failed to process PDF: ${error.message}`)
  }
}

async function initializePinecone() {
  if (isInitialized) return

  try {
    const index = pinecone.Index(process.env.PINECONE_INDEX_NAME!)
    
    console.log('Processing PDF...')
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
  } catch (error) {
    console.error('Initialization error:', error)
    throw error
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
  } catch (error) {
    console.error('Error in POST handler:', error)
    return NextResponse.json({ 
      error: 'Failed to process request: ' + (error.message || 'Unknown error')
    }, { status: 500 })
  }
}
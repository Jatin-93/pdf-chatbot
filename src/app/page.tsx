'use client'

import { useState, useEffect } from 'react'

export default function Home() {
  const [query, setQuery] = useState('')
  const [answer, setAnswer] = useState('')
  const [loading, setLoading] = useState(false)
  const [mounted, setMounted] = useState(false)

  // Handle mounting state to prevent hydration mismatch
  useEffect(() => {
    setMounted(true)
  }, [])

  // Don't render content until after mount to prevent hydration issues
  if (!mounted) {
    return null
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return

    try {
      setLoading(true)
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: query.trim() })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setAnswer(data.answer || 'No answer found.')
    } catch (error) {
      console.error('Error:', error)
      setAnswer('Sorry, something went wrong. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="min-h-screen bg-gray-50 py-12 px-4">
      <div className="max-w-3xl mx-auto">
        <h1 className="text-3xl font-bold text-center mb-8">
          Ask Questions About the Book
        </h1>
        
        <form onSubmit={handleSubmit} className="space-y-4 mb-8">
          <div>
            <label htmlFor="query" className="sr-only">
              Your question
            </label>
            <input
              id="query"
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask a question about the book..."
              className="w-full p-3 border border-gray-300 rounded-md shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              disabled={loading}
            />
          </div>
          
          <button
            type="submit"
            disabled={!query.trim() || loading}
            className={`w-full py-3 px-6 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors
              ${loading 
                ? 'bg-gray-400 cursor-not-allowed' 
                : 'bg-blue-600 hover:bg-blue-700 text-white'
              }`}
          >
            {loading ? (
              <span className="flex items-center justify-center">
                <span className="mr-2">Processing</span>
                <span className="animate-pulse">...</span>
              </span>
            ) : (
              'Ask Question'
            )}
          </button>
        </form>

        {answer && (
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-lg font-semibold mb-2">Answer:</h2>
            <div className="prose max-w-none text-gray-700 whitespace-pre-wrap">
              {answer}
            </div>
          </div>
        )}
      </div>
    </main>
  )
}
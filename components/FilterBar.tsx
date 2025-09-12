import { useState } from 'react'
import { Filter, ChevronDown } from 'lucide-react'

interface FilterOption {
  value: string
  label: string
}

interface FilterBarProps {
  filters: {
    [key: string]: FilterOption[]
  }
  onFilterChange: (filterKey: string, value: string) => void
  className?: string
}

export default function FilterBar({ filters, onFilterChange, className = "" }: FilterBarProps) {
  const [openFilters, setOpenFilters] = useState<{ [key: string]: boolean }>({})

  const toggleFilter = (filterKey: string) => {
    setOpenFilters(prev => ({
      ...prev,
      [filterKey]: !prev[filterKey]
    }))
  }

  return (
    <div className={`flex flex-wrap gap-4 ${className}`}>
      {Object.entries(filters).map(([filterKey, options]) => (
        <div key={filterKey} className="relative">
          <button
            onClick={() => toggleFilter(filterKey)}
            className="flex items-center gap-2 px-4 py-2 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
          >
            <Filter className="w-4 h-4" />
            <span className="capitalize">{filterKey}</span>
            <ChevronDown className={`w-4 h-4 transition-transform ${openFilters[filterKey] ? 'rotate-180' : ''}`} />
          </button>
          
          {openFilters[filterKey] && (
            <div className="absolute top-full left-0 mt-1 w-48 bg-white border border-gray-300 rounded-lg shadow-lg z-10">
              <div className="p-2">
                <button
                  onClick={() => {
                    onFilterChange(filterKey, '')
                    toggleFilter(filterKey)
                  }}
                  className="w-full text-left px-3 py-2 text-sm hover:bg-gray-100 rounded"
                >
                  Todos
                </button>
                {options.map((option) => (
                  <button
                    key={option.value}
                    onClick={() => {
                      onFilterChange(filterKey, option.value)
                      toggleFilter(filterKey)
                    }}
                    className="w-full text-left px-3 py-2 text-sm hover:bg-gray-100 rounded"
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

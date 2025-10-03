import { NextApiRequest, NextApiResponse } from 'next'
import { PrismaClient } from '@prisma/client'
import { spawn } from 'child_process'
import path from 'path'

const prisma = new PrismaClient()

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Método no permitido' })
  }

  const { analysisType, algorithm = 'lr', scriptName } = req.body

  try {
    // Determinar el script a ejecutar
    let pythonScript: string
    let args: string[]
    
    if (analysisType === 'comparison') {
      pythonScript = path.join(process.cwd(), 'ml_comparison_five_algorithms.py')
      args = []
    } else if (algorithm === 'knn') {
      pythonScript = path.join(process.cwd(), 'ml_knn_analysis.py')
      args = ['--type', analysisType]
    } else if (algorithm === 'nb') {
      pythonScript = path.join(process.cwd(), 'ml_naive_bayes_analysis.py')
      args = ['--type', analysisType]
    } else if (algorithm === 'kmeans') {
      pythonScript = path.join(process.cwd(), 'ml_kmeans_analysis.py')
      args = ['--type', analysisType]
            } else if (algorithm === 'tree') {
              pythonScript = path.join(process.cwd(), 'ml_decision_tree_analysis.py')
              args = ['--type', analysisType]
            } else if (algorithm === 'arima') {
              pythonScript = path.join(process.cwd(), 'ml_arima_analysis.py')
              args = ['--type', analysisType]
            } else if (algorithm === 'exponential') {
              pythonScript = path.join(process.cwd(), 'ml_exponential_smoothing_analysis.py')
              args = ['--type', analysisType]
            } else if (algorithm === 'randomforest') {
              pythonScript = path.join(process.cwd(), 'ml_random_forest_analysis.py')
              args = ['--type', analysisType]
            } else if (algorithm === 'xgboost') {
              pythonScript = path.join(process.cwd(), 'ml_xgboost_analysis.py')
              args = ['--type', analysisType]
            } else if (algorithm === 'neural') {
              pythonScript = path.join(process.cwd(), 'ml_neural_network_analysis.py')
              args = ['--type', analysisType]
            } else {
              pythonScript = path.join(process.cwd(), 'ml_specific_analysis.py')
              args = ['--type', analysisType]
            }
    
    const pythonProcess = spawn('python3', [pythonScript, ...args], {
      cwd: process.cwd(),
      env: { ...process.env, DATABASE_URL: process.env.DATABASE_URL }
    })

    let output = ''
    let errorOutput = ''

    pythonProcess.stdout.on('data', (data) => {
      output += data.toString()
    })

    pythonProcess.stderr.on('data', (data) => {
      errorOutput += data.toString()
    })

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        res.status(200).json({
          success: true,
          analysisType,
          algorithm,
          output: output.trim(),
          message: `Análisis ${analysisType} con ${algorithm.toUpperCase()} completado exitosamente`
        })
      } else {
        res.status(500).json({
          success: false,
          error: errorOutput,
          message: `Error ejecutando análisis ${analysisType}`
        })
      }
    })

  } catch (error) {
    console.error('Error en análisis ML:', error)
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Error desconocido',
      message: 'Error interno del servidor'
    })
  }
}

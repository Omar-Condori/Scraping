import { NextApiRequest, NextApiResponse } from 'next'
import { spawn } from 'child_process'
import path from 'path'
import fs from 'fs'

// Cargar el sistema de predicción
let predictionSystem = null

// Función para cargar el sistema de predicción
async function loadPredictionSystem() {
  if (predictionSystem) return predictionSystem
  
  try {
    const { spawn } = require('child_process')
    const pythonScript = path.join(process.cwd(), 'real_time_prediction_system.py')
    
    // Ejecutar script de Python para cargar el sistema
    const pythonProcess = spawn('python3', [pythonScript, '--load-only'], {
      cwd: process.cwd(),
      env: { ...process.env, DATABASE_URL: process.env.DATABASE_URL }
    })
    
    return new Promise((resolve, reject) => {
      pythonProcess.on('close', (code) => {
        if (code === 0) {
          predictionSystem = { loaded: true }
          resolve(predictionSystem)
        } else {
          reject(new Error('Error cargando sistema de predicción'))
        }
      })
    })
  } catch (error) {
    throw new Error(`Error inicializando sistema: ${error.message}`)
  }
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Método no permitido' })
  }

  const { action, ...data } = req.body

  try {
    // Cargar sistema de predicción si no está cargado
    await loadPredictionSystem()

    let pythonScript: string
    let args: string[]

    switch (action) {
      case 'predict':
        // Predicción individual
        pythonScript = path.join(process.cwd(), 'predict_single.py')
        args = [
          '--title', data.title || '',
          '--content', data.content || '',
          '--description', data.description || '',
          '--return-probabilities', data.return_probabilities ? 'true' : 'false'
        ]
        break

      case 'predict_batch':
        // Predicción en lote
        pythonScript = path.join(process.cwd(), 'predict_batch.py')
        args = [
          '--articles', JSON.stringify(data.articles || []),
          '--return-probabilities', data.return_probabilities ? 'true' : 'false'
        ]
        break

      case 'predict_recent':
        // Predicción de artículos recientes
        pythonScript = path.join(process.cwd(), 'predict_recent.py')
        args = [
          '--limit', (data.limit || 10).toString(),
          '--return-probabilities', data.return_probabilities ? 'true' : 'false'
        ]
        break

      case 'compare':
        // Comparar predicciones con categorías reales
        pythonScript = path.join(process.cwd(), 'compare_predictions.py')
        args = ['--limit', (data.limit || 20).toString()]
        break

      case 'stats':
        // Obtener estadísticas
        pythonScript = path.join(process.cwd(), 'get_prediction_stats.py')
        args = []
        break

      default:
        return res.status(400).json({ 
          success: false, 
          error: 'Acción no válida',
          message: 'Las acciones válidas son: predict, predict_batch, predict_recent, compare, stats'
        })
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
        try {
          const result = JSON.parse(output.trim())
          res.status(200).json({
            success: true,
            action,
            data: result,
            message: `Predicción ${action} completada exitosamente`
          })
        } catch (parseError) {
          res.status(200).json({
            success: true,
            action,
            output: output.trim(),
            message: `Predicción ${action} completada exitosamente`
          })
        }
      } else {
        res.status(500).json({
          success: false,
          error: errorOutput,
          message: `Error ejecutando predicción ${action}`
        })
      }
    })

  } catch (error: any) {
    console.error(`Error en la API de predicción:`, error)
    res.status(500).json({ 
      success: false, 
      error: error.message, 
      message: 'Error interno del servidor' 
    })
  }
}


import React, { useState, useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Loader2, Zap, ArrowRight, TrendingUp, Cpu } from 'lucide-react';

// Mock function representing the machine learning model inference.
// UPDATED: Now includes conceptual logic for Task Rank and Correlation.
const runAiInference = (taskData) => {
    // We are simulating the model's prediction based on the input features.

    const points = parseInt(taskData.points, 10);
    const length = parseInt(taskData.length, 10);
    const deadline = parseInt(taskData.deadline, 10);
    const rank = parseInt(taskData.rank, 10); // New
    const correlation = parseFloat(taskData.correlation); // New

    // Baseline performance ratio (e.g., the average in your data)
    let predictedRatio = 0.55;

    // Conceptual AI Logic:

    // 1. Points Impact (Strong positive correlation)
    predictedRatio += (points - 200) * 0.0008;

    // 2. Length Impact (Negative correlation)
    predictedRatio -= (length - 5) * 0.03;

    // 3. Deadline Impact (Inverse: Shorter deadline means better focus/higher ratio)
    predictedRatio += (30 - deadline) * 0.005;

    // 4. Variable Length Condition Impact (Conceptual: if variable length, perhaps more complex)
    if (taskData.variableLength === '1') {
        predictedRatio -= 0.05;
    }

    // 5. Task Rank Impact (Conceptual: Lower rank (1) means highest priority, increasing ratio)
    // Rank 1 is best, Rank 6 is worst.
    predictedRatio += (3 - rank) * 0.02;

    // 6. Correlation Impact (Conceptual: Higher correlation (positive) between points and length might be easier to manage, increasing ratio)
    predictedRatio += correlation * 0.05;

    // Clip the ratio between 0 and 1
    predictedRatio = Math.min(1.0, Math.max(0.0, predictedRatio));

    return predictedRatio;
};


const App = () => {
    const [taskInput, setTaskInput] = useState({
        length: '5',
        deadline: '10',
        points: '250',
        rank: '2', // Initial state for new field
        correlation: '0.0', // Initial state for new field
        variableLength: '0',
    });
    const [prediction, setPrediction] = useState(null);
    const [isLoading, setIsLoading] = useState(false);

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setTaskInput(prev => ({ ...prev, [name]: value }));
    };

    const handlePrioritize = () => {
        setIsLoading(true);
        setPrediction(null);

        // Input validation for correlation
        if (isNaN(parseFloat(taskInput.correlation))) {
            alert("Correlation must be a number between -1.0 and 1.0.");
            setIsLoading(false);
            return;
        }

        // Simulate network/computation delay
        setTimeout(() => {
            const result = runAiInference(taskInput);
            setPrediction(result);
            setIsLoading(false);
        }, 1500);
    };

    const predictionData = useMemo(() => {
        if (prediction === null) return [];
        return [
            { name: 'Predicted Ratio', ratio: parseFloat(prediction.toFixed(4)) },
            { name: 'Unearned Ratio', ratio: parseFloat((1 - prediction).toFixed(4)) },
        ];
    }, [prediction]);

    const getRatioColor = (ratio) => {
        if (ratio >= 0.8) return 'bg-green-500';
        if (ratio >= 0.6) return 'bg-yellow-500';
        return 'bg-red-500';
    };

    const getRatioText = (ratio) => {
        if (ratio >= 0.8) return 'Excellent Performance Expected';
        if (ratio >= 0.6) return 'Good Performance Expected';
        if (ratio >= 0.4) return 'Average Performance Expected';
        return 'Low Performance Expected';
    };

    return (
        <div className="min-h-screen bg-gray-50 p-4 sm:p-8 font-[Inter]">
            <header className="text-center mb-8">
                <h1 className="text-4xl font-extrabold text-blue-800 flex items-center justify-center space-x-3">
                    <Zap className="w-8 h-8 text-yellow-500"/>
                    <span>AI Task Prioritization Simulator</span>
                </h1>
                <p className="text-gray-600 mt-2">Simulate performance based on task characteristics (Conceptually powered by a Random Forest Model trained on your project data).</p>
            </header>

            <div className="max-w-4xl mx-auto bg-white p-6 md:p-10 rounded-xl shadow-2xl border border-gray-100">
                <div className="grid md:grid-cols-2 gap-8">
                    {/* --- Input Panel --- */}
                    <div className="bg-blue-50 p-6 rounded-lg shadow-inner">
                        <h2 className="text-2xl font-semibold text-blue-700 mb-4 flex items-center">
                            <Cpu className="w-5 h-5 mr-2"/>
                            Task Parameters
                        </h2>
                        <div className="space-y-4">
                            {/* Task Points */}
                            <div>
                                <label className="block text-sm font-medium text-gray-700">Task Points (Reward)</label>
                                <input
                                    type="number"
                                    name="points"
                                    value={taskInput.points}
                                    onChange={handleInputChange}
                                    min="100"
                                    max="500"
                                    className="mt-1 block w-full rounded-md border-gray-300 shadow-sm p-2.5 focus:border-blue-500 focus:ring-blue-500"
                                />
                                <p className="text-xs text-gray-500 mt-1">Value: 100 - 500 (High points = High Incentive)</p>
                            </div>

                            {/* Task Length (Turns) */}
                            <div>
                                <label className="block text-sm font-medium text-gray-700">Task Length (Turns/Effort)</label>
                                <input
                                    type="number"
                                    name="length"
                                    value={taskInput.length}
                                    onChange={handleInputChange}
                                    min="1"
                                    max="10"
                                    className="mt-1 block w-full rounded-md border-gray-300 shadow-sm p-2.5 focus:border-blue-500 focus:ring-blue-500"
                                />
                                <p className="text-xs text-gray-500 mt-1">Length: 1 - 10 (High length = High Effort)</p>
                            </div>

                            {/* Task Deadline (Turns Remaining) */}
                            <div>
                                <label className="block text-sm font-medium text-gray-700">Task Deadline (Turns)</label>
                                <input
                                    type="number"
                                    name="deadline"
                                    value={taskInput.deadline}
                                    onChange={handleInputChange}
                                    min="5"
                                    max="30"
                                    className="mt-1 block w-full rounded-md border-gray-300 shadow-sm p-2.5 focus:border-blue-500 focus:ring-blue-500"
                                />
                                <p className="text-xs text-gray-500 mt-1">Deadline: 5 - 30 (Low number = High Urgency)</p>
                            </div>

                            {/* Task Rank (New Field) */}
                            <div>
                                <label className="block text-sm font-medium text-gray-700">Task Rank (1=Highest Priority, 6=Lowest)</label>
                                <input
                                    type="number"
                                    name="rank"
                                    value={taskInput.rank}
                                    onChange={handleInputChange}
                                    min="1"
                                    max="6"
                                    className="mt-1 block w-full rounded-md border-gray-300 shadow-sm p-2.5 focus:border-blue-500 focus:ring-blue-500"
                                />
                                <p className="text-xs text-gray-500 mt-1">Simulates the inherent priority given to the task.</p>
                            </div>

                            {/* Correlation (New Field) */}
                            <div>
                                <label className="block text-sm font-medium text-gray-700">Correlation (Points-Length)</label>
                                <input
                                    type="number"
                                    name="correlation"
                                    value={taskInput.correlation}
                                    onChange={handleInputChange}
                                    step="0.1"
                                    min="-1.0"
                                    max="1.0"
                                    className="mt-1 block w-full rounded-md border-gray-300 shadow-sm p-2.5 focus:border-blue-500 focus:ring-blue-500"
                                />
                                <p className="text-xs text-gray-500 mt-1">The correlation between points and length for this task set (-1.0 to 1.0).</p>
                            </div>

                            {/* Variable Length Condition */}
                            <div>
                                <label className="block text-sm font-medium text-gray-700">Variable Length Task?</label>
                                <select
                                    name="variableLength"
                                    value={taskInput.variableLength}
                                    onChange={handleInputChange}
                                    className="mt-1 block w-full rounded-md border-gray-300 shadow-sm p-2.5 focus:border-blue-500 focus:ring-blue-500"
                                >
                                    <option value="0">0 - Fixed Length</option>
                                    <option value="1">1 - Variable Length</option>
                                </select>
                            </div>
                        </div>

                        <button
                            onClick={handlePrioritize}
                            disabled={isLoading}
                            className="mt-8 flex items-center justify-center space-x-2 bg-purple-600 hover:bg-purple-700 text-white font-bold py-3 px-6 rounded-lg shadow-md transition duration-300 disabled:opacity-50 disabled:cursor-not-allowed w-full"
                        >
                            {isLoading ? (
                                <Loader2 className="w-5 h-5 animate-spin" />
                            ) : (
                                <Zap className="w-5 h-5"/>
                            )}
                            <span>{isLoading ? 'Running Model...' : 'Predict Performance'}</span>
                        </button>
                    </div>

                    {/* --- Output Panel --- */}
                    <div className="p-6 rounded-lg border border-gray-200">
                        <h2 className="text-2xl font-semibold text-gray-800 mb-4 flex items-center">
                            <TrendingUp className="w-5 h-5 mr-2"/>
                            Prediction Output
                        </h2>

                        {prediction === null && !isLoading && (
                            <div className="text-center p-8 bg-gray-50 rounded-lg text-gray-500">
                                <ArrowRight className="w-8 h-8 mx-auto mb-2"/>
                                <p>Enter task parameters and click 'Predict Performance' to see results.</p>
                            </div>
                        )}

                        {isLoading && (
                            <div className="text-center p-8 bg-yellow-50 rounded-lg text-yellow-700">
                                <Loader2 className="w-8 h-8 mx-auto mb-2 animate-spin"/>
                                <p>Running AI model... Simulating prediction from your tuned Random Forest Regressor.</p>
                            </div>
                        )}

                        {prediction !== null && (
                            <div className="space-y-4">
                                {/* Gauge/Metric */}
                                <div className="text-center">
                                    <div className={`p-6 rounded-full inline-block ${getRatioColor(prediction)} text-white font-extrabold text-3xl shadow-xl transition-all duration-500`}>
                                        {(prediction * 100).toFixed(1)}%
                                    </div>
                                    <p className="text-lg font-bold text-gray-800 mt-3">
                                        Predicted Performance Ratio
                                    </p>
                                    <p className="text-md text-gray-600">
                                        {getRatioText(prediction)}
                                    </p>
                                </div>

                                {/* Bar Chart */}
                                <div className="h-48 w-full">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <BarChart
                                            data={predictionData}
                                            layout="vertical"
                                            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                                        >
                                            <CartesianGrid strokeDasharray="3 3" vertical={false} />
                                            <XAxis type="number" domain={[0, 1]} tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
                                            <YAxis dataKey="name" type="category" stroke="#555" />
                                            <Tooltip formatter={(value) => [`${(value * 100).toFixed(1)}%`, 'Ratio']} />
                                            <Bar dataKey="ratio" fill="#27ae60" radius={[4, 4, 4, 4]} />
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>

                                {/* Action Recommendation */}
                                <div className={`p-4 rounded-lg text-white font-semibold shadow-md ${prediction >= 0.7 ? 'bg-green-600' : 'bg-orange-500'}`}>
                                    {prediction >= 0.7 ?
                                        'Recommendation: HIGH PRIORITY - This task aligns well with optimal performance factors. Focus on efficiency.' :
                                        'Recommendation: MEDIUM/LOW PRIORITY - This task presents performance challenges. Consider re-prioritizing or improving strategy.'
                                    }
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            <footer className="text-center text-gray-500 text-sm mt-8">
                <p>AI-Task-Management-System-Project | Built with React and Tailwind CSS</p>
                <p>Simulation uses conceptual logic derived from EarnedPoints/OptimalPoints regression analysis.</p>
            </footer>
        </div>
    );
};

export default App;

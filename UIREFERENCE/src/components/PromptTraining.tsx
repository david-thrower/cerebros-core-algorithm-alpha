import React, { useState } from 'react';
import { Trash2Icon, CheckIcon, Edit2Icon } from 'lucide-react';
interface PromptExample {
  id: number;
  prompt: string;
  reasoning: string;
  output: string;
}
export function PromptTraining() {
  const [examples, setExamples] = useState<PromptExample[]>([{
    id: 1,
    prompt: 'Write a professional paragraph about cloud computing benefits for a business proposal',
    reasoning: 'Focus on business value, cost efficiency, and scalability. Use formal tone.',
    output: 'Cloud computing offers transformative advantages for modern enterprises. By leveraging scalable infrastructure and pay-as-you-go models, organizations can significantly reduce capital expenditure while maintaining operational flexibility. This technology enables seamless collaboration across distributed teams and provides robust disaster recovery capabilities.'
  }, {
    id: 2,
    prompt: 'Create a casual introduction paragraph for a tech blog post about AI',
    reasoning: 'Use conversational tone, make it engaging and accessible. Avoid jargon.',
    output: "Hey there! Let's talk about artificial intelligence - and no, we're not diving into scary robot takeover scenarios. AI is actually becoming a pretty cool tool that's making our daily lives easier in ways you might not even notice. From your phone's autocorrect to Netflix recommendations, it's everywhere."
  }, {
    id: 3,
    prompt: 'Generate a descriptive paragraph about a mountain landscape for travel content',
    reasoning: 'Use vivid imagery, sensory details, and evocative language to create atmosphere.',
    output: 'The jagged peaks pierce through wisps of morning clouds, their snow-capped summits gleaming in the golden sunlight. Ancient pine forests cascade down the slopes like emerald waterfalls, while crystal-clear streams carve their way through valleys below. The crisp mountain air carries the scent of wildflowers and distant adventure.'
  }]);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editedExample, setEditedExample] = useState<PromptExample | null>(null);
  const handleEdit = (example: PromptExample) => {
    setEditingId(example.id);
    setEditedExample({
      ...example
    });
  };
  const handleSave = (id: number) => {
    if (editedExample) {
      setExamples(examples.map(ex => ex.id === id ? editedExample : ex));
    }
    setEditingId(null);
    setEditedExample(null);
  };
  const handleCancel = () => {
    setEditingId(null);
    setEditedExample(null);
  };
  const handleDelete = (id: number) => {
    setExamples(examples.filter(ex => ex.id !== id));
  };
  const updateEditedField = (field: keyof PromptExample, value: string) => {
    if (editedExample) {
      setEditedExample({
        ...editedExample,
        [field]: value
      });
    }
  };
  return <div className="w-full min-h-screen bg-gray-50 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <button className="flex items-center text-gray-600 hover:text-gray-900 mb-6">
          <span className="mr-2">←</span>
          Back to Dashboard
        </button>
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Create Your AI Assistant
          </h1>
          <p className="text-gray-600">
            Follow these steps to personalize your assistant with your unique
            work style
          </p>
        </div>
        {/* Step Navigation */}
        <div className="flex items-center justify-between mb-12">
          {[{
          num: 1,
          label: 'Work Products',
          sublabel: 'Upload example documents'
        }, {
          num: 2,
          label: 'Review Data',
          sublabel: 'Add training examples'
        }, {
          num: 3,
          label: 'Communications',
          sublabel: 'Upload conversations'
        }, {
          num: 4,
          label: 'References',
          sublabel: 'Add knowledge base'
        }, {
          num: 5,
          label: 'Training',
          sublabel: 'Build your assistant'
        }].map((step, idx) => <div key={step.num} className="flex items-center flex-1">
              <div className="flex flex-col items-center">
                <div className={`w-12 h-12 rounded-full flex items-center justify-center text-white font-semibold ${step.num === 2 ? 'bg-blue-600' : 'bg-gray-300'}`}>
                  {step.num}
                </div>
                <div className="mt-2 text-center">
                  <div className="font-semibold text-sm text-gray-900">
                    {step.label}
                  </div>
                  <div className="text-xs text-gray-500">{step.sublabel}</div>
                </div>
              </div>
              {idx < 4 && <div className="flex-1 h-0.5 bg-gray-300 mx-4 mt-[-40px]" />}
            </div>)}
        </div>
        {/* Main Content */}
        <div className="bg-white rounded-lg shadow-sm p-8">
          <div className="mb-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-2">
              Review Data
            </h2>
            <p className="text-gray-600">
              Add example prompts and their expected responses. This helps train
              your AI assistant to understand your preferences.
            </p>
          </div>
          {/* Column Headers */}
          <div className="grid grid-cols-12 gap-4 mb-4">
            <div className="col-span-4">
              <h3 className="font-semibold text-gray-900">Example Prompts</h3>
            </div>
            <div className="col-span-4">
              <h3 className="font-semibold text-gray-900">
                Example Reasoning Path
              </h3>
            </div>
            <div className="col-span-4">
              <h3 className="font-semibold text-gray-900">
                Example Final Product
              </h3>
            </div>
          </div>
          {/* Example Rows */}
          <div className="space-y-4">
            {examples.map((example, index) => {
            const isEditing = editingId === example.id;
            const displayExample = isEditing && editedExample ? editedExample : example;
            return <div key={example.id} className="grid grid-cols-12 gap-4 p-4 border border-gray-200 rounded-lg">
                  {/* Prompt Column */}
                  <div className="col-span-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-gray-700">
                        Example {index + 1}
                      </span>
                    </div>
                    {isEditing ? <textarea value={displayExample.prompt} onChange={e => updateEditedField('prompt', e.target.value)} className="w-full p-2 border border-gray-300 rounded text-sm" rows={4} /> : <p className="text-sm text-gray-600">
                        {displayExample.prompt}
                      </p>}
                  </div>
                  {/* Reasoning Column */}
                  <div className="col-span-4">
                    <div className="mb-2">
                      <span className="text-sm font-medium text-gray-700 invisible">
                        Placeholder
                      </span>
                    </div>
                    {isEditing ? <textarea value={displayExample.reasoning} onChange={e => updateEditedField('reasoning', e.target.value)} className="w-full p-2 border border-gray-300 rounded text-sm bg-blue-50" rows={4} /> : <div className="bg-blue-50 p-2 rounded">
                        <p className="text-sm text-gray-600">
                          {displayExample.reasoning}
                        </p>
                      </div>}
                  </div>
                  {/* Output Column */}
                  <div className="col-span-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-gray-700 invisible">
                        Placeholder
                      </span>
                      <div className="flex gap-2">
                        {isEditing ? <>
                            <button onClick={() => handleSave(example.id)} className="text-green-600 hover:text-green-700" title="Save">
                              <CheckIcon className="w-4 h-4" />
                            </button>
                            <button onClick={handleCancel} className="text-gray-600 hover:text-gray-700" title="Cancel">
                              ✕
                            </button>
                          </> : <>
                            <button onClick={() => handleEdit(example)} className="text-blue-600 hover:text-blue-700" title="Edit">
                              <Edit2Icon className="w-4 h-4" />
                            </button>
                            <button onClick={() => handleDelete(example.id)} className="text-red-600 hover:text-red-700" title="Delete">
                              <Trash2Icon className="w-4 h-4" />
                            </button>
                          </>}
                      </div>
                    </div>
                    {isEditing ? <textarea value={displayExample.output} onChange={e => updateEditedField('output', e.target.value)} className="w-full p-2 border border-gray-300 rounded text-sm bg-green-50" rows={4} /> : <div className="bg-green-50 p-2 rounded">
                        <p className="text-sm text-gray-700">
                          {displayExample.output}
                        </p>
                      </div>}
                  </div>
                </div>;
          })}
          </div>
          {/* Footer */}
          <div className="flex justify-between items-center mt-8 pt-6 border-t border-gray-200">
            <button className="px-6 py-2 text-gray-600 hover:text-gray-900">
              Back
            </button>
            <button className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
              Continue
            </button>
          </div>
        </div>
      </div>
    </div>;
}
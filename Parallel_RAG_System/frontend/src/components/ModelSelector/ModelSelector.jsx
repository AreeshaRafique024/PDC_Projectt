/**
 * ModelSelector Component
 * Dropdown for selecting LLM models
 */

import React, { useState, useEffect, useRef } from "react";
import { ChevronDown, CheckCircle, XCircle, Sparkles } from "lucide-react";
import "./ModelSelector.css";

const ModelSelector = ({
  models = [],
  selectedModel,
  onSelectModel,
  loading = false,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const [groupBy, setGroupBy] = useState("category"); // 'category' or 'provider'
  const dropdownRef = useRef(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Filter models based on search term
  const filteredModels = models.filter((model) =>
    model.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Group models
  const groupedModels = () => {
    if (groupBy === "category") {
      const categories = {};
      filteredModels.forEach((model) => {
        const category = model.category || "Other";
        if (!categories[category]) {
          categories[category] = [];
        }
        categories[category].push(model);
      });
      return categories;
    } else {
      const providers = {};
      filteredModels.forEach((model) => {
        const provider = model.provider || "Other";
        if (!providers[provider]) {
          providers[provider] = [];
        }
        providers[provider].push(model);
      });
      return providers;
    }
  };

  const handleSelectModel = (model) => {
    if (model.available) {
      onSelectModel(model);
      setIsOpen(false);
      setSearchTerm("");
    }
  };

  const selectedModelData = models.find((m) => m.id === selectedModel);

  return (
    <div className="model-selector" ref={dropdownRef}>
      <button
        className={`model-selector-button ${isOpen ? "active" : ""}`}
        onClick={() => setIsOpen(!isOpen)}
        disabled={loading}
      >
        <div className="button-content">
          <Sparkles className="icon sparkle" />
          <div className="button-text">
            <span className="label">Selected Model</span>
            <span className="value">
              {selectedModelData ? selectedModelData.name : "Choose a model..."}
            </span>
          </div>
        </div>
        <ChevronDown className={`icon chevron ${isOpen ? "rotated" : ""}`} />
      </button>

      {isOpen && (
        <div className="model-selector-dropdown">
          <div className="dropdown-header">
            <input
              type="text"
              placeholder="Search models..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="search-input"
              autoFocus
            />
            <div className="group-toggle">
              <button
                className={groupBy === "category" ? "active" : ""}
                onClick={() => setGroupBy("category")}
              >
                Category
              </button>
              <button
                className={groupBy === "provider" ? "active" : ""}
                onClick={() => setGroupBy("provider")}
              >
                Provider
              </button>
            </div>
          </div>

          <div className="dropdown-content">
            {Object.entries(groupedModels()).map(([group, groupModels]) => (
              <div key={group} className="model-group">
                <div className="group-title">{group}</div>
                {groupModels.length === 0 ? (
                  <div className="no-models">No models found</div>
                ) : (
                  groupModels.map((model) => (
                    <button
                      key={model.id}
                      className={`model-item ${
                        selectedModel === model.id ? "selected" : ""
                      } ${!model.available ? "disabled" : ""}`}
                      onClick={() => handleSelectModel(model)}
                      disabled={!model.available}
                    >
                      <div className="model-info">
                        <span className="model-name">{model.name}</span>
                        <span className="model-provider">{model.provider}</span>
                      </div>
                      <div className="model-status">
                        {model.available ? (
                          <CheckCircle className="icon available" />
                        ) : (
                          <XCircle className="icon unavailable" />
                        )}
                      </div>
                    </button>
                  ))
                )}
              </div>
            ))}
          </div>

          <div className="dropdown-footer">
            <div className="status-legend">
              <div className="legend-item">
                <CheckCircle className="icon available" />
                <span>Available</span>
              </div>
              <div className="legend-item">
                <XCircle className="icon unavailable" />
                <span>API Key Required</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelSelector;

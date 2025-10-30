"""Energy use case functions refactored from instances/energy/executive.py."""

import pandas as pd
import copy
import numpy as np
from typing import Optional
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class EnergyFunctions:
    """Energy use case functions with dependencies injected via constructor."""
    
    def __init__(
        self,
        model,
        dataset: pd.DataFrame,
        y_values: pd.Series,
        explainer,
        dice_exp,
        dice_dataset: pd.DataFrame,
    ):
        """
        Initialize energy functions with required dependencies.
        
        Args:
            model: Trained model for energy consumption prediction
            dataset: Dataset DataFrame (without y column)
            y_values: Target variable Series
            explainer: SHAP explainer instance
            dice_exp: DiCE explainer instance
            dice_dataset: Dataset for DiCE (without prediction column)
        """
        self.model = model
        self.dataset = dataset
        self.y_values = y_values
        self.explainer = explainer
        self.dice_exp = dice_exp
        self.dice_dataset = dice_dataset
    
    def available_functions(self) -> str:
        """Return a description of available functions."""
        text = "<p>Here are the available functions:</p>"
        text += "<ul>"
        text += "<li><code>count_all()</code>: Count all instances in the dataset.</li>"
        text += "<li><code>count_group()</code>: Count instances in a group, filtered by available features.</li>"
        text += "<li><code>show_ids()</code>: Show all available IDs.</li>"
        text += "<li><code>show_one()</code>: Show data for a specific ID.</li>"
        text += "<li><code>show_group()</code>: Show data for a group, filtered by available features.</li>"
        text += "<li><code>predict_one()</code>: Predict energy consumption for a specific ID.</li>"
        text += "<li><code>predict_group()</code>: Predict energy consumption for a group, filtered by available features.</li>"
        text += "<li><code>predict_new()</code>: Predict energy consumption for new data.</li>"
        text += "<li><code>mistake_one()</code>: Show prediction error for a specific ID.</li>"
        text += "<li><code>mistake_group()</code>: Show prediction errors for a group, filtered by available features.</li>"
        text += "<li><code>explain_one()</code>: Explain prediction for a specific ID using SHAP values.</li>"
        text += "<li><code>explain_group()</code>: Explain predictions for a group using SHAP values, filtered by available features.</li>"
        text += "<li><code>cfes_one()</code>: Generate counterfactual explanations for a specific ID using DiCE.</li>"
        text += "<li><code>what_if_one()</code>: Show what-if analysis for a specific ID.</li>"
        text += "</ul>"
        text += "<p>Let me know if you want to use any of these functions.</p>"
        return text
    
    def about_dataset(self) -> str:
        """Provide information about the dataset."""
        text = "<p>The dataset <b>summer_workday_test</b> is a test dataset contains information about energy consumption.</p>"
        text += f"<p>It has <var>{len(self.dataset)}</var> instances and <var>{len(self.dataset.columns)}</var> features.</p>"
        text += f"<p>The features are: {', '.join([f'<code>{feature}</code>' for feature in self.dataset.columns])}.</p>"
        text += "<p>I can also give you the more in depth statistics of the dataset. Would you like to see it?</p>"
        return text
    
    def about_dataset_in_depth(self) -> str:
        """Provide detailed statistics about the dataset."""
        text = "<p>Here are the statistics of each feature in the dataset:</p>"
        text += f"<p>{self.dataset.describe().round(2).to_html()}</p>"
        return text
    
    def about_model(self) -> str:
        """Provide information about the model."""
        text = "<p>The model is a <b>genetic programming symbolic regressor</b>.</p>"
        text += "<p>It was trained on the dataset features to predict energy consumption.</p>"
        text += "<p>Symbolic regression is a machine learning technique that aims to identify an underlying mathematical expression that best describes a relationship.</p>"
        text += "<p>It begins by building a population of naive random formulas to represent a relationship between known independent variables and their dependent variable targets in order to predict new data.</p>"
        text += "<p>Each successive generation of programs is then evolved from the one that came before it by selecting the fittest individuals from the population to undergo genetic operations.<p>"
        
        # Access model._program if available
        program_str = getattr(self.model, '_program', 'N/A')
        text += f"<p>The program that is currently loaded looks like this: <code>{program_str}</code></p>"
        text += f"<p>Where features are: {' '.join([f'X{i} = {self.dataset.columns[i]}' for i in range(len(self.dataset.columns))])}</p>"
        return text
    
    def model_accuracy(self) -> str:
        """Calculate and return model accuracy metrics."""
        pred = self.model.predict(self.dataset)
        explained_variance = explained_variance_score(self.y_values, pred)
        rmse = mean_squared_error(self.y_values, pred, squared=False)
        mae = mean_absolute_error(self.y_values, pred)
        text = f"<p>The model has an <b>explained variance score</b> of <var>{explained_variance:.2f}</var>.</p>"
        text += f"<p>The <b>root mean squared error</b> of the model is <var>{rmse:.2f}</var>.</p>"
        text += f"<p>The <b>mean absolute error</b> of the model is <var>{mae:.2f}</var>.</p>"
        return text
    
    def about_explainer(self) -> str:
        """Provide information about the explainer."""
        text = "<p>The explainer is a SHAP Kernel Explainer.</p>"
        text += "<p>SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model. \
            It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions.</p>"
        return text
    
    def count_all(self) -> str:
        """Count all instances in the dataset."""
        return f"<p>There are <var>{len(self.dataset)}</var> instances in the dataset.</p>"
    
    def count_group(
        self,
        indoor_temperature_min: Optional[float] = None,
        indoor_temperature_max: Optional[float] = None,
        outdoor_temperature_min: Optional[float] = None,
        outdoor_temperature_max: Optional[float] = None,
        past_electricity_min: Optional[float] = None,
        past_electricity_max: Optional[float] = None,
    ) -> str:
        """Count instances in a filtered group."""
        intro = self._format_group(
            indoor_temperature_min, indoor_temperature_max,
            outdoor_temperature_min, outdoor_temperature_max,
            past_electricity_min, past_electricity_max
        )
        result = self.dataset.copy()
        
        if indoor_temperature_min is not None:
            result = result[result['indoor_temperature'] > indoor_temperature_min]
        if indoor_temperature_max is not None:
            result = result[result['indoor_temperature'] < indoor_temperature_max]
        if outdoor_temperature_min is not None:
            result = result[result['outdoor_temperature'] > outdoor_temperature_min]
        if outdoor_temperature_max is not None:
            result = result[result['outdoor_temperature'] < outdoor_temperature_max]
        if past_electricity_min is not None:
            result = result[result['past_electricity'] > past_electricity_min]
        if past_electricity_max is not None:
            result = result[result['past_electricity'] < past_electricity_max]
        
        if result.empty:
            return "<p>There is no data for the selected group.</p>"
        
        return intro + f"<p>There are <var>{len(result)}</var> instances in the selected group.</p>"
    
    def show_ids(self) -> str:
        """Show all available IDs."""
        ids = self.dataset.index.get_level_values(0).unique().sort_values()
        ids_str = ', '.join([f'<var>{id}</var>' for id in ids])
        return f"<p>Available <code>ID</code> values are: {ids_str}.</p>"
    
    def show_one(self, id) -> str:
        """Show data for a specific ID."""
        if id not in self.dataset.index:
            return f"<p>There is no data for <code>ID</code> <var>{id}</var>.</p>"
        intro = f"<p>Here is the data for <code>ID</code> <var>{id}</var>:</p>"
        renamed = self.dataset.rename(columns={
            'outdoor_temperature': 'outdoor temperature',
            'indoor_temperature': 'indoor temperature',
            'past_electricity': 'past electricity'
        })
        framed = pd.DataFrame(renamed)
        table = f"<p>{framed.loc[id].to_frame().T.to_html()}</p>"
        return intro + table
    
    def show_group(
        self,
        indoor_temperature_min: Optional[float] = None,
        indoor_temperature_max: Optional[float] = None,
        outdoor_temperature_min: Optional[float] = None,
        outdoor_temperature_max: Optional[float] = None,
        past_electricity_min: Optional[float] = None,
        past_electricity_max: Optional[float] = None,
    ) -> str:
        """Show data for a filtered group."""
        intro = self._format_group(
            indoor_temperature_min, indoor_temperature_max,
            outdoor_temperature_min, outdoor_temperature_max,
            past_electricity_min, past_electricity_max
        )
        result = self.dataset.copy()
        
        if indoor_temperature_min is not None:
            result = result[result['indoor_temperature'] > indoor_temperature_min]
        if indoor_temperature_max is not None:
            result = result[result['indoor_temperature'] < indoor_temperature_max]
        if outdoor_temperature_min is not None:
            result = result[result['outdoor_temperature'] > outdoor_temperature_min]
        if outdoor_temperature_max is not None:
            result = result[result['outdoor_temperature'] < outdoor_temperature_max]
        if past_electricity_min is not None:
            result = result[result['past_electricity'] > past_electricity_min]
        if past_electricity_max is not None:
            result = result[result['past_electricity'] < past_electricity_max]
        
        if result.empty:
            return "<p>There is no data for the selected group.</p>"
        
        intro += "<p>Showing the data for the selected group.</p>"
        renamed = result.rename(columns={
            'outdoor_temperature': 'outdoor temperature',
            'indoor_temperature': 'indoor temperature',
            'past_electricity': 'past electricity'
        })
        renamed.sort_index(inplace=True)
        table = f"<p>{renamed.to_html()}</p>"
        return intro + table
    
    def predict_one(self, id) -> str:
        """Predict energy consumption for a specific ID."""
        if id not in self.dataset.index:
            return f"<p>There is no data for <code>ID</code> <var>{id}</var>.</p>"
        data = self.dataset.loc[id].to_frame().T
        prediction = self.model.predict(data)
        rounded = round(prediction[0], 2)
        return f"<p>The prediction for <code>ID</code> <var>{id}</var> is <samp>{rounded}</samp>.</p>"
    
    def predict_group(
        self,
        indoor_temperature_min: Optional[float] = None,
        indoor_temperature_max: Optional[float] = None,
        outdoor_temperature_min: Optional[float] = None,
        outdoor_temperature_max: Optional[float] = None,
        past_electricity_min: Optional[float] = None,
        past_electricity_max: Optional[float] = None,
    ) -> str:
        """Predict energy consumption for a filtered group."""
        intro = self._format_group(
            indoor_temperature_min, indoor_temperature_max,
            outdoor_temperature_min, outdoor_temperature_max,
            past_electricity_min, past_electricity_max
        )
        
        data = self.dataset.copy()
        if indoor_temperature_min is not None:
            data = data[data['indoor_temperature'] > indoor_temperature_min]
        if indoor_temperature_max is not None:
            data = data[data['indoor_temperature'] < indoor_temperature_max]
        if outdoor_temperature_min is not None:
            data = data[data['outdoor_temperature'] > outdoor_temperature_min]
        if outdoor_temperature_max is not None:
            data = data[data['outdoor_temperature'] < outdoor_temperature_max]
        if past_electricity_min is not None:
            data = data[data['past_electricity'] > past_electricity_min]
        if past_electricity_max is not None:
            data = data[data['past_electricity'] < past_electricity_max]
        
        if data.empty:
            return "<p>There is no data for the selected group.</p>"
        
        intro += "<p>Here are the predictions for the selected group.</p>"
        prediction = self.model.predict(data)
        framed = pd.DataFrame(prediction, columns=['prediction'], index=data.index)
        framed.sort_index(inplace=True)
        table = f"<p>{framed.to_html()}</p>"
        return intro + table
    
    def predict_new(self, indoor_temperature: float, outdoor_temperature: float, past_electricity: float) -> str:
        """Predict energy consumption for new data."""
        data = pd.DataFrame([[indoor_temperature, outdoor_temperature, past_electricity]],
                           columns=['indoor_temperature', 'outdoor_temperature', 'past_electricity'])
        prediction = self.model.predict(data)
        rounded = round(prediction[0], 2)
        text = "<p>Let's consider a new data sample with the following features:</p>"
        text += f"<p><ul>{''.join([f'<li><code>{feature}</code> = <var>{self._extract_value(value)}</var></li>' for feature, value in data.to_dict().items()])}</ul><p>"
        text += f"<p>The model prediction for the new data will be <samp>{rounded}</samp>.</p>"
        return text
    
    def mistake_one(self, id) -> str:
        """Show prediction error for a specific ID."""
        if id not in self.dataset.index:
            return f"<p>There is no data for <code>ID</code> <var>{id}</var>.</p>"
        data = self.dataset.loc[id].to_frame().T
        prediction = self.model.predict(data)[0]
        rounded = round(prediction, 2)
        actual = self.y_values.loc[id]
        text = f"<p>The prediction for <code>ID</code> <var>{id}</var> is <samp>{rounded}</samp>.</p>"
        text += f"<p>The actual value is <samp>{actual}</samp>.</p>"
        text += f"<p>The error is <samp>{round(abs(actual - prediction), 2)}</samp>.</p>"
        return text
    
    def mistake_group(
        self,
        indoor_temperature_min: Optional[float] = None,
        indoor_temperature_max: Optional[float] = None,
        outdoor_temperature_min: Optional[float] = None,
        outdoor_temperature_max: Optional[float] = None,
        past_electricity_min: Optional[float] = None,
        past_electricity_max: Optional[float] = None,
    ) -> str:
        """Show prediction errors for a filtered group."""
        intro = self._format_group(
            indoor_temperature_min, indoor_temperature_max,
            outdoor_temperature_min, outdoor_temperature_max,
            past_electricity_min, past_electricity_max
        )
        
        data = self.dataset.copy()
        if indoor_temperature_min is not None:
            data = data[data['indoor_temperature'] > indoor_temperature_min]
        if indoor_temperature_max is not None:
            data = data[data['indoor_temperature'] < indoor_temperature_max]
        if outdoor_temperature_min is not None:
            data = data[data['outdoor_temperature'] > outdoor_temperature_min]
        if outdoor_temperature_max is not None:
            data = data[data['outdoor_temperature'] < outdoor_temperature_max]
        if past_electricity_min is not None:
            data = data[data['past_electricity'] > past_electricity_min]
        if past_electricity_max is not None:
            data = data[data['past_electricity'] < past_electricity_max]
        
        if data.empty:
            return "<p>There is no data for the selected group.</p>"
        
        intro += "<p>Here are the errors for the selected group.</p>"
        prediction = self.model.predict(data)
        labels = self.y_values.loc[data.index].values
        errors = abs(labels - prediction).round(2)
        framed = pd.DataFrame(errors, columns=['error'], index=data.index)
        framed['prediction'] = prediction.round(2)
        framed['actual'] = labels
        framed = framed[['actual', 'prediction', 'error']]
        framed.sort_index(inplace=True)
        table = f"<p>{framed.to_html()}</p>"
        return intro + table
    
    def explain_one(self, id) -> str:
        """Explain prediction for a specific ID using SHAP values."""
        if id not in self.dataset.index:
            return f"<p>There is no data for <code>ID</code> <var>{id}</var>.</p>"
        data = self.dataset.loc[id].to_frame().T
        shap_values = self.explainer.shap_values(data, nsamples=10_000, silent=True)
        influences = shap_values.squeeze()
        result = pd.DataFrame(influences, columns=['influence'], index=self.dataset.columns).sort_values(
            by='influence', key=abs, ascending=False
        )
        text = f"<p>For the instance with <code>ID</code> <var>{id}</var> the feature importances are:</p>"
        text += f"<p>{result.to_html()}</p>"
        return text
    
    def explain_group(
        self,
        indoor_temperature_min: Optional[float] = None,
        indoor_temperature_max: Optional[float] = None,
        outdoor_temperature_min: Optional[float] = None,
        outdoor_temperature_max: Optional[float] = None,
        past_electricity_min: Optional[float] = None,
        past_electricity_max: Optional[float] = None,
    ) -> str:
        """Explain predictions for a filtered group using SHAP values."""
        intro = self._format_group(
            indoor_temperature_min, indoor_temperature_max,
            outdoor_temperature_min, outdoor_temperature_max,
            past_electricity_min, past_electricity_max
        )
        
        data = self.dataset.copy()
        if indoor_temperature_min is not None:
            data = data[data['indoor_temperature'] > indoor_temperature_min]
        if indoor_temperature_max is not None:
            data = data[data['indoor_temperature'] < indoor_temperature_max]
        if outdoor_temperature_min is not None:
            data = data[data['outdoor_temperature'] > outdoor_temperature_min]
        if outdoor_temperature_max is not None:
            data = data[data['outdoor_temperature'] < outdoor_temperature_max]
        if past_electricity_min is not None:
            data = data[data['past_electricity'] > past_electricity_min]
        if past_electricity_max is not None:
            data = data[data['past_electricity'] < past_electricity_max]
        
        if data.empty:
            return "<p>There is no data for the selected group.</p>"
        
        intro += "<p>Here are the feature importances for the selected group.</p>"
        result = pd.DataFrame(index=data.index, columns=[])
        
        for idx in data.index:
            shap_values = self.explainer.shap_values(data.loc[idx], nsamples=10_000, silent=True)
            influences = shap_values.squeeze()
            influences = pd.DataFrame(influences, columns=['influence'], index=self.dataset.columns).sort_values(
                by='influence', key=abs, ascending=False
            )
            for feature in influences.index:
                result.loc[idx, f"influence of {feature}"] = influences.loc[feature, 'influence']
        
        result.sort_index(inplace=True)
        table = f"<p>{result.to_html()}</p>"
        return intro + table
    
    def cfes_one(self, id) -> str:
        """Generate counterfactual explanations for a specific ID using DiCE."""
        if id not in self.dataset.index:
            return f"<p>There is no data for <code>ID</code> <var>{id}</var>.</p>"
        
        original_prediction = self.model.predict(self.dice_dataset.loc[[id]])[0]
        
        cfe = self.dice_exp.generate_counterfactuals(
            self.dice_dataset.loc[[id]],
            total_CFs=10,
            desired_range=[0, original_prediction],
            features_to_vary=['indoor_temperature', 'outdoor_temperature']
        )
        
        final_cfes = cfe.cf_examples_list[0].final_cfs_df
        final_cfe_ids = list(final_cfes.index)
        if 'prediction' in final_cfes.columns:
            final_cfes.pop('prediction')
        
        new_predictions = self.model.predict(final_cfes)
        original_instance = self.dice_dataset.loc[[id]]
        
        output_string = f"<p>The original prediction for the data sample with <code>ID</code> <var>{id}</var> is <samp>{str(round(original_prediction, 2))}</samp>.</p>"
        output_string += "<p>Here are some options to change the prediction of this instance."
        output_string += "<p>First, if you"
        transition_words = ["Further,", "Also,", "In addition,", "Furthermore,"]
        
        for i, c_id in enumerate(final_cfe_ids):
            if i < 3 and i < len(final_cfe_ids):
                if i != 0:
                    output_string += f"<p>{np.random.choice(transition_words)} if you"
                output_string += self._get_change_string(final_cfes.loc[[c_id]], original_instance)
                new_prediction = str(round(new_predictions[i], 2))
                output_string += f", the model will predict <samp>{new_prediction}</samp>.</p>"
        
        return output_string
    
    def what_if_one(
        self,
        id,
        indoor_temperature: Optional[float] = None,
        outdoor_temperature: Optional[float] = None,
        past_electricity: Optional[float] = None,
    ) -> str:
        """Show what-if analysis for a specific ID."""
        if id not in self.dataset.index:
            return f"<p>There is no data for <code>ID</code> <var>{id}</var>.</p>"
        
        original_instance = self.dice_dataset.loc[[id]]
        original_prediction = self.model.predict(original_instance)[0]
        if isinstance(original_prediction, dict):
            original_prediction = list(original_prediction.values())[0]
        
        changed_instance = original_instance.copy()
        if indoor_temperature is not None:
            changed_instance['indoor_temperature'] = indoor_temperature
        if outdoor_temperature is not None:
            changed_instance['outdoor_temperature'] = outdoor_temperature
        if past_electricity is not None:
            changed_instance['past_electricity'] = past_electricity
        
        new_prediction = self.model.predict(changed_instance)[0]
        if isinstance(new_prediction, dict):
            new_prediction = list(new_prediction.values())[0]
        
        text = f"<p>For the data sample with <code>ID</code> <var>{id}</var>, the original features are:</p>"
        text += f"<ul>{''.join([f'<li><code>{feature}</code> = <var>{self._extract_value(value)}</var></li>' for feature, value in original_instance.to_dict().items()])}</ul>"
        text += f"<p>The model predicts <samp>{str(round(original_prediction, 2))}</samp> for this instance.</p>"
        text += f"<p>Let's change the features to: <ul>{''.join([f'<li><code>{feature}</code> = <var>{self._extract_value(value)}</var></li>' for feature, value in changed_instance.to_dict().items()])}</ul></p>"
        text += f"<p>Then the model will predict <samp>{str(round(new_prediction, 2))}</samp>.</p>"
        return text
    
    # Helper methods
    
    def _format_group(
        self,
        indoor_temperature_min: Optional[float] = None,
        indoor_temperature_max: Optional[float] = None,
        outdoor_temperature_min: Optional[float] = None,
        outdoor_temperature_max: Optional[float] = None,
        past_electricity_min: Optional[float] = None,
        past_electricity_max: Optional[float] = None,
    ) -> str:
        """Format group filtering description."""
        text = "<p>Grouping the data as follows:<ul>"
        if indoor_temperature_min is not None:
            text += f"<li><code>indoor temperature</code> is more than <var>{indoor_temperature_min}</var></li>"
        if indoor_temperature_max is not None:
            text += f"<li><code>indoor temperature</code> is less than <var>{indoor_temperature_max}</var></li>"
        if outdoor_temperature_min is not None:
            text += f"<li><code>outdoor temperature</code> is more than <var>{outdoor_temperature_min}</var></li>"
        if outdoor_temperature_max is not None:
            text += f"<li><code>outdoor temperature</code> is less than <var>{outdoor_temperature_max}</var></li>"
        if past_electricity_min is not None:
            text += f"<li><code>past electricity</code> is more than <var>{past_electricity_min}</var></li>"
        if past_electricity_max is not None:
            text += f"<li><code>past electricity</code> is less than <var>{past_electricity_max}</var></li>"
        text += "</ul></p>"
        return text
    
    def _get_change_string(self, cfe: pd.DataFrame, original_instance: pd.DataFrame) -> str:
        """Build a string describing changes between counterfactual and original instance."""
        cfe_features = list(cfe.columns)
        original_features = list(original_instance.columns)
        assert set(cfe_features) == set(original_features), "CFE features and Original Instance features are different!"
        
        change_string = ""
        for feature in cfe_features:
            orig_f = original_instance[feature].values[0]
            cfe_f = cfe[feature].values[0]
            
            if isinstance(cfe_f, str):
                cfe_f = float(cfe_f)
            
            if orig_f != cfe_f:
                if cfe_f > orig_f:
                    inc_dec = " increase"
                else:
                    inc_dec = " decrease"
                change_string += f"{inc_dec} <code>{feature}</code> to <var>{str(round(cfe_f, 2))}</var>"
                change_string += " and "
        
        # Strip off last " and "
        if change_string.endswith(" and "):
            change_string = change_string[:-5]
        return change_string
    
    def _extract_value(self, value) -> float:
        """Extract numeric value if input is a dictionary with an index."""
        if isinstance(value, dict):
            return list(value.values())[0]
        return value


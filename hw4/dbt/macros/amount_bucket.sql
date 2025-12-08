{% macro amount_bucket(amount_column) -%}
    case
        when {{ amount_column }} >= {{ var('transaction_amount_threshold_high') }} then 'HIGH'
        when {{ amount_column }} >= {{ var('transaction_amount_threshold_medium') }} then 'MEDIUM'
        else 'LOW'
    end
{%- endmacro %}

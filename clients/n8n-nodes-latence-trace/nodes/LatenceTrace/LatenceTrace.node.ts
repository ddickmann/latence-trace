import {
	IExecuteFunctions,
	INodeExecutionData,
	INodeType,
	INodeTypeDescription,
	NodeOperationError,
} from 'n8n-workflow';

export class LatenceTrace implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Latence TRACE',
		name: 'latenceTrace',
		icon: 'file:latence.svg',
		group: ['transform'],
		version: 1,
		subtitle: '={{$parameter["operation"]}}',
		description: 'Score groundedness and redact PII through the TRACE compliance runtime.',
		defaults: {
			name: 'Latence TRACE',
		},
		inputs: ['main'],
		outputs: ['main', 'main', 'main'],
		outputNames: ['green', 'amber', 'red'],
		credentials: [
			{
				name: 'latenceTraceApi',
				required: true,
			},
		],
		properties: [
			{
				displayName: 'Operation',
				name: 'operation',
				type: 'options',
				noDataExpression: true,
				options: [
					{
						name: 'Score Groundedness',
						value: 'score',
						description:
							'Send question + context + answer to TRACE and get back groundedness, band, and channel scores.',
						action: 'Score groundedness',
					},
					{
						name: 'Route by Band',
						value: 'route',
						description:
							'Score and route the item to the green / amber / red output branch.',
						action: 'Route by band',
					},
					{
						name: 'Redact Compliance PII',
						value: 'redactCompliance',
						description:
							'Detect and mask or replace PII with the TRACE compliance runtime.',
						action: 'Redact compliance PII',
					},
				],
				default: 'score',
			},
			{
				displayName: 'Question',
				name: 'question',
				type: 'string',
				default: '',
				required: true,
				description: 'The user question that produced the response.',
				displayOptions: {
					show: {
						operation: ['score', 'route'],
					},
				},
			},
			{
				displayName: 'Response Text',
				name: 'responseText',
				type: 'string',
				typeOptions: { rows: 4 },
				default: '',
				required: true,
				description: 'The generated answer to evaluate.',
				displayOptions: {
					show: {
						operation: ['score', 'route'],
					},
				},
			},
			{
				displayName: 'Raw Context',
				name: 'rawContext',
				type: 'string',
				typeOptions: { rows: 4 },
				default: '',
				required: true,
				description:
					'Retrieved evidence the answer must be grounded in. Concatenate chunks with blank lines.',
				displayOptions: {
					show: {
						operation: ['score', 'route'],
					},
				},
			},
			{
				displayName: 'Profile',
				name: 'profile',
				type: 'options',
				default: 'standard',
				options: [
					{ name: 'Standard', value: 'standard' },
					{ name: 'Quality', value: 'quality' },
					{ name: 'Code', value: 'code' },
				],
				description:
					'Scoring profile. Quality applies NLI aggregation; Code uses AST-aware pooling.',
				displayOptions: {
					show: {
						operation: ['score', 'route'],
					},
				},
			},
			{
				displayName: 'Text to Redact',
				name: 'complianceText',
				type: 'string',
				typeOptions: { rows: 6 },
				default: '',
				required: true,
				description: 'Text that may contain PII. Raw text is sent only to the compliance endpoint.',
				displayOptions: {
					show: {
						operation: ['redactCompliance'],
					},
				},
			},
			{
				displayName: 'Labels',
				name: 'complianceLabels',
				type: 'string',
				default: 'person,email,phone_number,date_of_birth,employee_id',
				description:
					'Optional comma-separated PII labels. Leave blank to use open mode with the full catalog.',
				displayOptions: {
					show: {
						operation: ['redactCompliance'],
					},
				},
			},
			{
				displayName: 'Redaction Mode',
				name: 'complianceRedactionMode',
				type: 'options',
				default: 'mask',
				options: [
					{ name: 'Mask', value: 'mask' },
					{ name: 'Replace', value: 'replace' },
				],
				description: 'Mask entities with label tokens or replace them with synthetic values.',
				displayOptions: {
					show: {
						operation: ['redactCompliance'],
					},
				},
			},
			{
				displayName: 'Tenant ID',
				name: 'tenantId',
				type: 'string',
				default: '',
				description:
					'Optional. Forwarded as X-Latence-Tenant-Id so per-tenant thresholds apply.',
			},
			{
				displayName: 'Timeout (ms)',
				name: 'timeoutMs',
				type: 'number',
				default: 60000,
				description: 'HTTP timeout for the TRACE call.',
			},
		],
	};

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const credentials = await this.getCredentials('latenceTraceApi');
		const baseUrl = ((credentials.baseUrl as string) || 'https://api.latence.ai').replace(/\/$/, '');

		const greenOut: INodeExecutionData[] = [];
		const amberOut: INodeExecutionData[] = [];
		const redOut: INodeExecutionData[] = [];
		const items = this.getInputData();

		for (let i = 0; i < items.length; i++) {
			const operation = this.getNodeParameter('operation', i) as string;
			const question = this.getNodeParameter('question', i) as string;
			const responseText = this.getNodeParameter('responseText', i) as string;
			const rawContext = this.getNodeParameter('rawContext', i) as string;
			const profile = this.getNodeParameter('profile', i) as string;
			const tenantId = this.getNodeParameter('tenantId', i) as string;
			const timeoutMs = this.getNodeParameter('timeoutMs', i) as number;

			const body = {
				question,
				response_text: responseText,
				raw_context: rawContext,
				profile,
			};

			const headers: Record<string, string> = {
				'content-type': 'application/json',
				accept: 'application/json',
			};
			if (tenantId) {
				headers['x-latence-tenant-id'] = tenantId;
			}

			let response;
			try {
				if (operation === 'redactCompliance') {
					const text = this.getNodeParameter('complianceText', i) as string;
					const rawLabels = this.getNodeParameter('complianceLabels', i) as string;
					const redactionMode = this.getNodeParameter('complianceRedactionMode', i) as string;
					const labels = rawLabels
						.split(',')
						.map((label) => label.trim())
						.filter(Boolean);
					response = await this.helpers.httpRequestWithAuthentication.call(this, 'latenceTraceApi', {
						method: 'POST',
						url: `${baseUrl}/v1/compliance/redact`,
						headers,
						json: true,
						body: {
							text,
							mode: labels.length ? 'category' : 'open',
							labels: labels.length ? labels : undefined,
							redact: true,
							redaction_mode: redactionMode,
							include_original_text: false,
						},
						timeout: timeoutMs,
					});
					greenOut.push({
						json: {
							...items[i].json,
							latence_compliance: response,
							redacted_text: response?.redacted_text,
							entity_count: response?.entity_count,
						},
						pairedItem: { item: i },
					});
					continue;
				}
				response = await this.helpers.httpRequestWithAuthentication.call(this, 'latenceTraceApi', {
					method: 'POST',
					url: `${baseUrl}/v1/score/groundedness`,
					headers,
					json: true,
					body,
					timeout: timeoutMs,
				});
			} catch (error) {
				throw new NodeOperationError(this.getNode(), (error as Error).message, {
					itemIndex: i,
				});
			}

			const band = String(response?.band ?? 'amber').toLowerCase();
			const enriched: INodeExecutionData = {
				json: {
					...items[i].json,
					latence_trace: response,
					latence_band: band,
				},
				pairedItem: { item: i },
			};

			if (operation === 'route') {
				if (band === 'green') greenOut.push(enriched);
				else if (band === 'red') redOut.push(enriched);
				else amberOut.push(enriched);
			} else {
				greenOut.push(enriched);
			}
		}

		return [greenOut, amberOut, redOut];
	}
}

import {
	IExecuteFunctions,
	INodeExecutionData,
	INodeType,
	INodeTypeDescription,
	NodeConnectionType,
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
		description: 'Score groundedness and route RAG answers by risk band.',
		defaults: {
			name: 'Latence TRACE',
		},
		inputs: [NodeConnectionType.Main],
		outputs: [NodeConnectionType.Main, NodeConnectionType.Main, NodeConnectionType.Main],
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
			},
			{
				displayName: 'Response Text',
				name: 'responseText',
				type: 'string',
				typeOptions: { rows: 4 },
				default: '',
				required: true,
				description: 'The generated answer to evaluate.',
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

# Single Sign-on (SSO)

Single Sign-on (SSO) allows your team members to authenticate to DataChain Studio using your organization's identity Provider (IdP) such as Okta, LDAP, Microsoft AD, etc.

We support integration with Okta, and instructions are provided below; but other IdPs should also work in a similar manner. If you need any support setting up your IdP integration, [let us know](../troubleshooting.md#support).

SSO for teams can be configured by team admins, and requires configuration on both DataChain Studio and the IdP. The exact steps for this depend on the IdP.

Once the SSO configuration is complete users can login to DataChain Studio by opening their team's login page `https://studio.datachain.ai/api/teams/<TEAM_NAME>/sso` in their browser. They can also login directly from their Okta end-user dashboards by clicking on the DataChain Studio integration icon.

If a user does not have a pre-assigned role when they sign in to a team, they will be auto-assigned the Viewer role.

## Okta integration

1. **Create Enterprise account**: SSO is available for DataChain Studio teams with enterprise account subscriptions. If you are on the Free or Basic plan of DataChain Studio, contact us to upgrade your account.

2. **Add integration with DataChain Studio in Okta**: Follow the instructions from the [Okta developer guide](https://developer.okta.com/docs/guides/build-sso-integration/saml2/main/#create-your-integration-in-okta). In short, login to Okta with an admin account, and follow these steps:
   1. In the admin console, go to `Applications` -> `Create App Integration` to create a private SSO integration.
   2. Use `SAML 2.0` as the `Sign in method` (and not `OIDC` or some other option).
   3. Enter any name (eg, `DataChain Studio`) as the `App name`.
   4. `Single sign-on URL`: [`https://studio.datachain.ai/api/teams/<TEAM_NAME>/saml/consume`](https://studio.datachain.ai/api/teams/<TEAM_NAME>/saml/consume) (Replace <TEAM_NAME> with the name of your team in Studio.
   5. `Audience URI (SP Entity ID)`: https://studio.datachain.ai/api/saml
   6. `Name ID Format`: Persistent
   7. `Application username (NameID)`: Okta username
   8. `Attribute Statements (optional)`:
      1. `Name`: email
      2. `Name format`: URI Reference
      3. `Value`: user.email

   Click on `Next` and `Finish`. Once the integration is created, open the `Sign On` tab and expand the `Hide Details` section. From here, copy the Identity Provider metadata URL.

3. **Configure DataChain Studio**: In your team settings, go to the `SSO` section and enable SSO. Enter the Identity Provider metadata URL that you copied from Okta.

4. **Assign users**: In Okta, assign users to the DataChain Studio application.

5. **Test the integration**: Users can now login to DataChain Studio using their Okta credentials by visiting the team's SSO login page.

## User roles

DataChain Studio supports the following user roles:

- **Admin**: Full access to team settings and all projects
- **Member**: Can create and manage projects, view all team projects
- **Viewer**: Read-only access to team projects

## Troubleshooting

If you encounter issues with SSO setup:

1. Verify that the URLs and configuration values are entered correctly
2. Check that users are properly assigned to the application in your IdP
3. Ensure that the Identity Provider metadata URL is accessible
4. Contact our support team if you need assistance with configuration

For more help, see our [troubleshooting guide](../troubleshooting.md).
